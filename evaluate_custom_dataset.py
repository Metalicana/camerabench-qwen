import os
import argparse
import torch
import json
import csv
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from sklearn.metrics import accuracy_score, average_precision_score
import numpy as np

# ---------- Qwen Setup ----------

def load_qwen(model_id: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


@torch.no_grad()
def get_model_answer(model, processor, image_1_path, image_2_path, question: str, options: dict):
    """
    Runs the Qwen model on two images and a prompt, returns model's best guess (A/B/C/D).
    """
    if not os.path.exists(image_1_path) or not os.path.exists(image_2_path):
        print(f"Missing one of the image paths: {image_1_path}, {image_2_path}")
        return None

    # Build chat message
    content = [
        {"type": "image", "image": image_1_path},
        {"type": "image", "image": image_2_path},
        {"type": "text", "text": f"{question}\nOptions:\nA. {options['A']}\nB. {options['B']}\nC. {options['C']}\nD. {options['D']}\nAnswer with only A, B, C, or D."}
    ]
    messages = [{"role": "user", "content": content}]
    
    # Tokenize + run inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
    
    output = model.generate(**inputs, max_new_tokens=5)
    decoded = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
    
    # Extract first valid letter
    for letter in ["A", "B", "C", "D"]:
        if letter in decoded:
            return letter
    return "Unknown"


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--json_file", required=True, help="Path to questions JSON file")
    ap.add_argument("--data_root", default="./")
    ap.add_argument("--out_file", default="results.csv")
    args = ap.parse_args()

    print(f"Loading model: {args.model}")
    model, processor = load_qwen(args.model)

    print(f"Loading questions from {args.json_file}")
    with open(args.json_file, "r") as f:
        questions = json.load(f)

    results_by_type = {}
    raw_rows = []

    for q in questions:
        qid = q["question_id"]
        qtype = q["question_type"]
        prompt = q["prompt"]
        correct = q["correct_answer"].strip()
        img1 = os.path.join(args.data_root, q["image_1"])
        img2 = os.path.join(args.data_root, q["image_2"])
        options = q["options"]

        print(f"Processing Q{qid} ({qtype})...")
        try:
            pred = get_model_answer(model, processor, img1, img2, prompt, options)
        except Exception as e:
            print(f"Skipped Q{qid} due to error: {e}")
            continue

        correct_flag = 1 if pred == correct else 0 if correct else "N/A"

        if qtype not in results_by_type:
            results_by_type[qtype] = {"correct": [], "total": 0}

        if correct_flag != "N/A":
            results_by_type[qtype]["correct"].append(correct_flag)
        results_by_type[qtype]["total"] += 1

        raw_rows.append([qid, qtype, pred, correct, correct_flag])

    # ---------- Save Results ----------

    with open(args.out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question_ID", "Type", "Predicted", "Correct", "Match"])
        writer.writerows(raw_rows)
    print(f"Raw results saved to {args.out_file}")

    # ---------- Summary ----------
    print("\n=== Accuracy by Category ===")
    for k, v in results_by_type.items():
        if not v["correct"]:
            acc = "N/A"
        else:
            acc = sum(v["correct"]) / len(v["correct"])
        print(f"{k:12s} : {acc}")
    print("Done.")


if __name__ == "__main__":
    main()
