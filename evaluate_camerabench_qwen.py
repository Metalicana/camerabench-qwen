# evaluate_camerabench_qwen.py
import os
import argparse
import pathlib
import requests
import torch
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from sklearn.metrics import average_precision_score
import imageio.v2 as imageio
from PIL import Image


# ---------- IO helpers ----------
def download_to(path, url, timeout=60):
    """Download a URL to a local path."""
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
    """Make H and W even to avoid codec quirks; pad last row/col if needed."""
    h, w = arr.shape[:2]
    pad_h = 1 if h % 2 != 0 else 0
    pad_w = 1 if w % 2 != 0 else 0
    if pad_h or pad_w:
        if arr.ndim == 3:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        else:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")
    return arr


def gif_to_mp4(gif_path: str, mp4_path: str, fps: int = 12):
    """
    Transcode a GIF to H.264 MP4, forcing RGB frames so channel count is consistent.
    Returns mp4_path if successful; raises if no frames could be written.
    """
    gif_path = str(gif_path)
    mp4_path = str(mp4_path)

    reader = imageio.get_reader(gif_path)
    meta = reader.get_meta_data()
    # Try to honor GIF fps if available
    out_fps = fps
    try:
        if "duration" in meta and meta["duration"]:
            # duration is per-frame milliseconds; fps ~ 1000/duration
            frame_ms = meta["duration"]
            if isinstance(frame_ms, (int, float)) and frame_ms > 0:
                out_fps = max(1, min(60, int(round(1000.0 / frame_ms))))
    except Exception:
        pass

    # robust writer: even dimensions, RGB frames
    writer = imageio.get_writer(
        mp4_path,
        fps=out_fps,
        codec="libx264",
        quality=8,
        macro_block_size=1  # avoids forced resizing to multiples of 16
    )

    frames_written = 0
    try:
        for frame in reader:
            # Normalize to RGB using PIL (handles L/P/LA/RGBA etc.)
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)
            img = Image.fromarray(frame)
            # drop alpha if present, convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.asarray(img)
            arr = _ensure_even_hw(arr)
            writer.append_data(arr)
            frames_written += 1
    finally:
        writer.close()
        reader.close()

    if frames_written == 0:
        raise RuntimeError("No valid frames found in GIF (could not transcode).")
    return mp4_path


# ---------- CameraBench taxonomy ----------
PRIMITIVES = [
    # Translation
    "dolly-in", "dolly-out", "pedestal-up", "pedestal-down", "truck-right", "truck-left",
    # Zoom
    "zoom-in", "zoom-out",
    # Rotation
    "pan-right", "pan-left", "tilt-up", "tilt-down", "roll-CW", "roll-CCW",
    # Static
    "no-motion",
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
    "Zooming":     ["zoom-in", "zoom-out"],
    "Rotation":    ["pan-right", "pan-left", "tilt-up", "tilt-down", "roll-CW", "roll-CCW"],
    "Static":      ["no-motion"],
}


# ---------- Qwen helpers ----------
def load_qwen(model_id: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


@torch.no_grad()
def yes_probability_on_video(model, processor, media_path: str, question: str) -> float:
    """
    Return P(yes) using the first generated token distribution.
    Fallback: generate a couple tokens and parse "yes"/"no".
    """
    device = next(model.parameters()).device

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
    ).to(device)

    # First-token probability path
    out = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        temperature=None,  # avoid irrelevant warning
    )
    first_logits = out.scores[0][0]  # (vocab,)
    probs = torch.softmax(first_logits, dim=-1)

    tok = processor.tokenizer
    vocab = tok.get_vocab()

    def to_ids(cands):
        return [tok.convert_tokens_to_ids(c) for c in cands if c in vocab]

    yes_ids = to_ids(["yes", "Yes", "YES"])
    no_ids = to_ids(["no", "No", "NO"])
    if yes_ids or no_ids:
        return float(probs[yes_ids].sum().item() if yes_ids else 0.0)

    # Fallback: short decode then check prefix
    seq = model.generate(**inputs, max_new_tokens=3, do_sample=False, temperature=None)
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
    ap.add_argument("--limit", type=int, default=0, help="limit number of videos; 0=all")
    ap.add_argument("--cache_dir", default="./cache_videos", help="where to store downloaded media")
    ap.add_argument("--gif_fps", type=int, default=12, help="fps used when transcoding GIF->MP4")
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

    processed = 0
    for i in range(n):
        row = ds[i]

        # Prefer public GIF URL; if present, download & transcode to MP4 once.
        gif_url = row.get("Video") or row.get("video") or row.get("video_gif") or row.get("gif")
        local_media = None
        try:
            if gif_url and str(gif_url).endswith(".gif"):
                gif_name = os.path.basename(str(gif_url).split("?")[0])
                gif_path = os.path.join(args.cache_dir, gif_name)
                if not os.path.exists(gif_path):
                    gif_path = download_to(gif_path, gif_url)
                mp4_name = os.path.splitext(gif_name)[0] + ".mp4"
                mp4_path = os.path.join(args.cache_dir, mp4_name)
                if not os.path.exists(mp4_path):
                    print(f"  transcoding GIF -> MP4: {gif_name} -> {mp4_name}")
                    gif_to_mp4(gif_path, mp4_path, fps=args.gif_fps)
                local_media = mp4_path
            else:
                # Try original MP4 path from repo (requires HF auth if gated)
                rel_path = row["path"]  # e.g., "videos/xxx.mp4"
                local_media = hf_hub_download(
                    repo_id="syCen/CameraBench",
                    filename=rel_path,
                    local_dir=args.cache_dir
                )
        except Exception as e:
            print(f"  WARN: sample {i} skipped (media issue): {e}")
            continue

        try:
            gt = set(row["labels"])
            for prim in PRIMITIVES:
                q = QUESTION_TEMPLATES[prim]
                p_yes = yes_probability_on_video(model, processor, local_media, q)
                all_scores[prim].append(p_yes)
                all_gts[prim].append(1 if prim in gt else 0)
            processed += 1
        except Exception as e:
            print(f"  WARN: sample {i} skipped (model/video decode): {e}")
            # keep arrays aligned by skipping appends for this sample
            # (we appended nothing for this sample, so no cleanup needed)
            continue

        if processed % 25 == 0:
            print(f"  processed {processed}/{n} successful")

    if processed == 0:
        print("No samples processed successfully. Check GIF→MP4 deps (pillow, imageio-ffmpeg) or HF auth for MP4s.")
        return

    # Compute AP per primitive
    per_class_ap = {}
    for prim in PRIMITIVES:
        y_true = all_gts[prim]
        y_score = all_scores[prim]
        # guard: if nothing collected for this label due to skips
        if len(y_true) == 0 or len(set(y_true)) < 2:
            per_class_ap[prim] = float("nan")
        else:
            per_class_ap[prim] = float(average_precision_score(y_true, y_score))

    # Macro AP across valid primitives
    valid_vals = [v for v in per_class_ap.values() if v == v]
    macro_ap = sum(valid_vals) / len(valid_vals) if valid_vals else float("nan")

    # Family AP
    family_ap = {}
    for fam, plist in FAMILIES.items():
        vals = [per_class_ap[p] for p in plist if per_class_ap[p] == per_class_ap[p]]
        family_ap[fam] = sum(vals) / len(vals) if vals else float("nan")

    # Pretty print
    print("\n=== Per-class AP ===")
    for k, v in per_class_ap.items():
        print(f"{k:12s} : {v:.4f}" if v == v else f"{k:12s} : NaN")

    print("\n=== Family AP ===")
    for k, v in family_ap.items():
        print(f"{k:12s} : {v:.4f}" if v == v else f"{k:12s} : NaN")

    print(f"\nMacro AP (all primitives): {macro_ap:.4f}" if macro_ap == macro_ap else "\nMacro AP: NaN")


if __name__ == "__main__":
    main()
