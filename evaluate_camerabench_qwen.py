# evaluate_camerabench_qwen.py
import os
import argparse
import pathlib, requests
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from sklearn.metrics import average_precision_score

def download_to(path, url):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            if chunk:
                f.write(chunk)
    return str(path)

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
    "Translation": ["dolly-in","dolly-out","pedestal-up","pedestal-down","truck-right","truck-left"],
    "Zooming":     ["zoom-in","zoom-out"],
    "Rotation":    ["pan-right","pan-left","tilt-up","tilt-down","roll-CW","roll-CCW"],
    "Static":      ["no-motion"],
}

def load_qwen(model_id: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

@torch.no_grad()
def yes_probability_on_video(model, processor, media_path: str, question: str) -> float:
    """
    Returns P(yes) from first generated token; falls back to short-text parsing.
    """
    device = next(model.parameters()).device
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": media_path},
            {"type": "text", "text": question}
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
    )
    first_logits = out.scores[0][0]  # (vocab,)
    probs = torch.softmax(first_logits, dim=-1)
    tok = processor.tokenizer
    vocab = tok.get_vocab()

    def to_ids(cands): 
        return [tok.convert_tokens_to_ids(c) for c in cands if c in vocab]

    yes_ids = to_ids(["yes","Yes","YES"])
    no_ids  = to_ids(["no","No","NO"])
    if yes_ids or no_ids:
        return float(probs[yes_ids].sum().item() if yes_ids else 0.0)

    # Fallback: short text generation then parse first word
    seq = model.generate(**inputs, max_new_tokens=3, do_sample=False)
    txt = processor.batch_decode(seq, skip_special_tokens=True)[0].strip().lower()
    if txt.startswith("yes"): return 1.0
    if txt.startswith("no"):  return 0.0
    return 0.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="limit number of videos; 0=all")
    ap.add_argument("--cache_dir", default="./cache_videos", help="where to store downloaded media")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    print("Loading dataset…")
    ds = load_dataset("syCen/CameraBench", split=args.split)

    print(f"Loading model: {args.model}")
    model, processor = load_qwen(args.model)

    all_scores = {k: [] for k in PRIMITIVES}
    all_gts    = {k: [] for k in PRIMITIVES}

    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    print(f"Evaluating {n} videos…")

    for i in range(n):
        row = ds[i]

        # Prefer public GIF URL if present; otherwise try MP4 via hf_hub (may require auth)
        gif_url = row.get("Video") or row.get("video") or row.get("video_gif") or row.get("gif")
        local_media = None
        try:
            if gif_url and str(gif_url).endswith(".gif"):
                fname = os.path.basename(str(gif_url).split("?")[0])
                local_media = download_to(os.path.join(args.cache_dir, fname), gif_url)
            else:
                rel_path = row["path"]  # e.g., "videos/xxx.mp4"
                local_media = hf_hub_download(
                    repo_id="syCen/CameraBench",
                    filename=rel_path,
                    local_dir=args.cache_dir
                )
        except Exception as e:
            print(f"  WARN: could not fetch media for sample {i}: {e}")
            continue

        gt = set(row["labels"])

        # Ask one yes/no per primitive
        for prim in PRIMITIVES:
            q = QUESTION_TEMPLATES[prim]
            p_yes = yes_probability_on_video(model, processor, local_media, q)
            all_scores[prim].append(p_yes)
            all_gts[prim].append(1 if prim in gt else 0)

        if (i + 1) % 25 == 0:
            print(f"  processed {i+1}/{n}")

    # Compute AP per primitive and aggregates
    per_class_ap = {}
    for prim in PRIMITIVES:
        y_true = all_gts[prim]
        y_score = all_scores[prim]
        if len(set(y_true)) < 2:
            per_class_ap[prim] = float("nan")
        else:
            per_class_ap[prim] = float(average_precision_score(y_true, y_score))

    macro_ap = sum(v for v in per_class_ap.values() if v == v) / \
               sum(1 for v in per_class_ap.values() if v == v)

    # Family AP
    family_ap = {}
    for fam, plist in FAMILIES.items():
        vals = [per_class_ap[p] for p in plist if per_class_ap[p] == per_class_ap[p]]
        family_ap[fam] = sum(vals)/len(vals) if vals else float("nan")

    # Pretty print
    print("\n=== Per-class AP ===")
    for k, v in per_class_ap.items():
        print(f"{k:12s} : {v:.4f}" if v == v else f"{k:12s} : NaN")

    print("\n=== Family AP ===")
    for k, v in family_ap.items():
        print(f"{k:12s} : {v:.4f}" if v == v else f"{k:12s} : NaN")

    print(f"\nMacro AP (all primitives): {macro_ap:.4f}")

if __name__ == "__main__":
    main()
