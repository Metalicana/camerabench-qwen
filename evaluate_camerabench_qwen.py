import os
import argparse
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from sklearn.metrics import average_precision_score

# ---------- Labels and Prompts ----------

PRIMITIVES = [
    "dolly-in", "dolly-out", "pedestal-up", "pedestal-down", "truck-right", "truck-left",
    "zoom-in", "zoom-out", "pan-right", "pan-left", "tilt-up", "tilt-down",
    "roll-CW", "roll-CCW", "no-motion"
]

QUESTION_TEMPLATES = {
    "dolly-in": "Does the camera move forward (a dolly-in)? Answer yes or no.",
    "dolly-out": "Does the camera move backward (a dolly-out)? Answer yes or no.",
    "pedestal-up": "Does the camera move straight up (a pedestal up)? Answer yes or no.",
    "pedestal-down": "Does the camera move straight down (a pedestal down)? Answer yes or no.",
    "truck-right": "Does the camera move sideways to the right (a truck right)? Answer yes or no.",
    "truck-left": "Does the camera move sideways to the left (a truck left)? Answer yes or no.",
    "zoom-in": "Does the camera zoom in (change focal length to magnify)? Answer yes or no.",
    "zoom-out": "Does the camera zoom out? Answer yes or no.",
    "pan-right": "Does the camera pan right (rotate to the right on the horizontal axis)? Answer yes or no.",
    "pan-left": "Does the camera pan left? Answer yes or no.",
    "tilt-up": "Does the camera tilt up (rotate upward)? Answer yes or no.",
    "tilt-down": "Does the camera tilt down? Answer yes or no.",
    "roll-CW": "Does the camera roll clockwise? Answer yes or no.",
    "roll-CCW": "Does the camera roll counterclockwise? Answer yes or no.",
    "no-motion": "Is the camera static with no intentional motion? Answer yes or no.",
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
        ]
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
    )
    first_logits = out.scores[0][0]
    probs = torch.softmax(first_logits, dim=-1)
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()

    yes_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["yes", "Yes", "YES"] if t in vocab]
    no_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["no", "No", "NO"] if t in vocab]

    if yes_ids or no_ids:
        return float(probs[yes_ids].sum().item()) if yes_ids else 0.0

    # fallback: decode text
    seq = model.generate(**inputs, max_new_tokens=3, do_sample=False)
    decoded = tokenizer.batch_decode(seq, skip_special_tokens=True)[0].strip().lower()
    if decoded.startswith("yes"):
        return 1.0
    elif decoded.startswith("no"):
        return 0.0
    return 0.5

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--local_data_root", default="./CameraBench")
    args = ap.parse_args()

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
        rel_path = row["path"]
        local_media = os.path.join(args.local_data_root, rel_path)

        if not os.path.exists(local_media):
            print(f" WARN: [{i}] Skipped (file missing): {local_media}")
            continue

        gt = set(row["labels"])
        try:
            for prim in PRIMITIVES:
                question = QUESTION_TEMPLATES[prim]
                score = yes_probability_on_video(model, processor, local_media, question)
                all_scores[prim].append(score)
                all_gts[prim].append(1 if prim in gt else 0)
        except Exception as e:
            print(f" WARN: [{i}] Skipped (decode/model): {e}")
            continue

    # ---------- Metrics ----------
    per_class_ap = {}
    for prim in PRIMITIVES:
        y_true = all_gts[prim]
        y_score = all_scores[prim]
        if len(y_true) == 0 or len(set(y_true)) < 2:
            per_class_ap[prim] = float("nan")
        else:
            per_class_ap[prim] = average_precision_score(y_true, y_score)

    valid = [v for v in per_class_ap.values() if not np.isnan(v)]
    macro_ap = sum(valid) / len(valid) if valid else float("nan")

    family_ap = {}
    for fam, plist in FAMILIES.items():
        vals = [per_class_ap[p] for p in plist if not np.isnan(per_class_ap[p])]
        family_ap[fam] = sum(vals) / len(vals) if vals else float("nan")

    print("\n=== Per-class AP ===")
    for k, v in per_class_ap.items():
        print(f"{k:12s} : {v:.4f}" if not np.isnan(v) else f"{k:12s} : NaN")

    print("\n=== Family AP ===")
    for k, v in family_ap.items():
        print(f"{k:12s} : {v:.4f}" if not np.isnan(v) else f"{k:12s} : NaN")

    print(f"\nMacro AP (all primitives): {macro_ap:.4f}" if not np.isnan(macro_ap) else "Macro AP: NaN")

if __name__ == "__main__":
    main()
