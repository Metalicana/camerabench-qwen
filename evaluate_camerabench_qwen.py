import os
import argparse
import pathlib
import requests
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from sklearn.metrics import average_precision_score
import imageio.v2 as imageio


# ---------- Helpers ----------
def download_to(path, url, timeout=60):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            if chunk:
                f.write(chunk)
    return str(path)


def _ensure_even_hw(arr):
    h, w = arr.shape[:2]
    pad_h = 1 if h % 2 != 0 else 0
    pad_w = 1 if w % 2 != 0 else 0
    if pad_h or pad_w:
        if arr.ndim == 3:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        else:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")
    return arr


def gif_to_mp4(gif_path: str, mp4_path: str, fps: int = 12, verbose: int = 1):
    """Transcode a GIF to H.264 MP4, forcing RGB frames and ensuring ≥2 frames."""
    reader = imageio.get_reader(gif_path)
    frames = []
    for frame in reader:
        img = Image.fromarray(frame).convert("RGB")
        arr = _ensure_even_hw(np.asarray(img))
        frames.append(arr)
    reader.close()

    if len(frames) < 2:
        raise RuntimeError(f"GIF has only {len(frames)} frame(s), skipping.")

    writer = imageio.get_writer(
        mp4_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    for f in frames:
        writer.append_data(f)
    writer.close()

    if verbose:
        print(f"    Transcoded {gif_path} → {mp4_path} with {len(frames)} frames")
    return mp4_path


# ---------- Labels and Prompts ----------
PRIMITIVES = [
    "dolly-in", "dolly-out", "pedestal-up", "pedestal-down",
    "truck-right", "truck-left",
    "zoom-in", "zoom-out",
    "pan-right", "pan-left",
    "tilt-up", "tilt-down",
    "roll-CW", "roll-CCW",
    "no-motion"
]

QUESTION_TEMPLATES = {
    "dolly-in":       "Does the camera move forward (a dolly-in)? Answer yes or no.",
    "dolly-out":      "Does the camera move backward (a dolly-out)? Answer yes or no.",
    "pedestal-up":    "Does the camera move straight up (a pedestal up)? Answer yes or no.",
    "pedestal-down":  "Does the camera move straight down (a pedestal down)? Answer yes or no.",
    "truck-right":    "Does the camera move sideways to the right (a truck right)? Answer yes or no.",
    "truck-left":     "Does the camera move sideways to the left (a truck left)? Answer yes or no.",
    "zoom-in":        "Does the camera zoom in (change focal length to magnify)? Answer yes or no.",
    "zoom-out":       "Does the camera zoom out? Answer yes or no.",
    "pan-right":      "Does the camera pan right (rotate to the right on the horizontal axis)? Answer yes or no.",
    "pan-left":       "Does the camera pan left? Answer yes or no.",
    "tilt-up":        "Does the camera tilt up (rotate upward)? Answer yes or no.",
    "tilt-down":      "Does the camera tilt down? Answer yes or no.",
    "roll-CW":        "Does the camera roll clockwise? Answer yes or no.",
    "roll-CCW":       "Does the camera roll counterclockwise? Answer yes or no.",
    "no-motion":      "Is the camera static with no intentional motion? Answer yes or no.",
}

FAMILIES = {
    "Translation": ["dolly-in", "dolly-out", "pedestal-up", "pedestal-down", "truck-right", "truck-left"],
    "Zooming": ["zoom-in", "zoom-out"],
    "Rotation": ["pan-right", "pan-left", "tilt-up", "tilt-down", "roll-CW", "roll-CCW"],
    "Static": ["no-motion"]
}


# ---------- Qwen Setup ----------
def load_qwen(model_id: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


@torch.no_grad()
def yes_probability_on_video(model, processor, media_path: str, question: str) -> float:
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": media_path},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        temperature=None
    )

    first_logits = out.scores[0][0]
    probs = torch.softmax(first_logits, dim=-1)
    tok = processor.tokenizer
    vocab = tok.get_vocab()
    to_ids = lambda cands: [tok.convert_tokens_to_ids(c) for c in cands if c in vocab]
    yes_ids = to_ids(["yes", "Yes", "YES"])
    no_ids = to_ids(["no", "No", "NO"])

    if yes_ids or no_ids:
        return float(probs[yes_ids].sum().item()) if yes_ids else 0.0

    # fallback
    seq = model.generate(**inputs, max_new_tokens=3, do_sample=False)
    txt = tok.batch_decode(seq, skip_special_tokens=True)[0].strip().lower()
    if txt.startswith("yes"):
        return 1.0
    if txt.startswith("no"):
        return 0.0
    return 0.5


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache_dir", default="./cache_videos")
    ap.add_argument("--gif_fps", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    print("Loading dataset…")
    ds = load_dataset("syCen/CameraBench", split=args.split)

    print(f"Loading model: {args.model}")
    model, processor = load_qwen(args.model)

    all_scores = {k: [] for k in PRIMITIVES}
    all_gts = {k: [] for k in PRIMITIVES}

    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    print(f"Evaluating {n} videos…")

    for i in range(n):
        row = ds[i]
        local_media = None
        try:
            gif_url = row.get("Video") or row.get("video") or row.get("video_gif")
            if gif_url and gif_url.endswith(".gif"):
                gif_name = os.path.basename(gif_url.split("?")[0])
                gif_path = os.path.join(args.cache_dir, gif_name)
                mp4_path = os.path.join(args.cache_dir, gif_name.replace(".gif", ".mp4"))

                if not os.path.exists(mp4_path):
                    if not os.path.exists(gif_path):
                        download_to(gif_path, gif_url)
                    print(f"  [{i}] Transcoding {gif_name}")
                    gif_to_mp4(gif_path, mp4_path, fps=args.gif_fps, verbose=1)

                local_media = mp4_path
            else:
                rel_path = row["path"]
                local_media = hf_hub_download(
                    repo_id="syCen/CameraBench",
                    filename=rel_path,
                    local_dir=args.cache_dir
                )
        except Exception as e:
            print(f"  WARN: [{i}] Skipped (fetch/transcode): {e}")
            continue

        gt = set(row["labels"])
        try:
            for prim in PRIMITIVES:
                q = QUESTION_TEMPLATES[prim]
                p_yes = yes_probability_on_video(model, processor, local_media, q)
                all_scores[prim].append(p_yes)
                all_gts[prim].append(1 if prim in gt else 0)
        except Exception as e:
            print(f"  WARN: [{i}] Skipped (decode/model): {e}")
            continue

    # === AP Metrics ===
    per_class_ap = {}
    for prim in PRIMITIVES:
        y_true, y_score = all_gts[prim], all_scores[prim]
        if len(y_true) == 0 or len(set(y_true)) < 2:
            per_class_ap[prim] = float("nan")
        else:
            per_class_ap[prim] = average_precision_score(y_true, y_score)

    valid = [v for v in per_class_ap.values() if v == v]
    macro_ap = sum(valid) / len(valid) if valid else float("nan")

    family_ap = {}
    for fam, plist in FAMILIES.items():
        vals = [per_class_ap[p] for p in plist if per_class_ap[p] == per_class_ap[p]]
        family_ap[fam] = sum(vals)/len(vals) if vals else float("nan")

    print("\n=== Per-class AP ===")
    for k, v in per_class_ap.items():
        print(f"{k:12s} : {v:.4f}" if v == v else f"{k:12s} : NaN")

    print("\n=== Family AP ===")
    for k, v in family_ap.items():
        print(f"{k:12s} : {v:.4f}" if v == v else f"{k:12s} : NaN")

    print(f"\nMacro AP (all primitives): {macro_ap:.4f}" if macro_ap == macro_ap else "Macro AP: NaN")


if __name__ == "__main__":
    main()
