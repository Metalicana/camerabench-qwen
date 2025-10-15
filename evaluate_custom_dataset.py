import os
import argparse
import torch
import json
import csv
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

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

    content = [
        {"type": "image", "image": image_1_path},
        {"type": "image", "image": image_2_path},
        {"type": "text", "text": f"{question}\nOptions:\nA. {options['A']}\nB. {options['B']}\nC. {options['C']}\nD. {options['D']}\nAnswer with only A, B, C, or D."}
    ]
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=5)
    decoded = processor.batch_decode(output, skip_special_tokens=True)[0].strip()

    for letter in ["A", "B", "C", "D"]:
        if letter in decoded:
            return letter
    return "Unknown"


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--json_file", required=True, help="Path to questions JSON file")
    ap.add_argument("--data_root", default="/path/to/codebase/Images", help="Base path for image files")
    ap.add_argument("--out_dir", default="results", help="Base output directory for results")
    args = ap.parse_args()

    print(f"Loading model: {args.model}")
    model, processor = load_qwen(args.model)

    print(f"Loading questions from {args.json_file}")
    with open(args.json_file, "r") as f:
        questions = json.load(f)

    all_results = []
    results_by_type = {}

    os.makedirs(args.out_dir, exist_ok=True)

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

        result_row = [qid, qtype, pred, correct, correct_flag]
        all_results.append(result_row)

        # Prepare per-type logging
        if qtype not in results_by_type:
            results_by_type[qtype] = {"rows": [], "correct": [], "total": 0}

        results_by_type[qtype]["rows"].append(result_row)
        if correct_flag != "N/A":
            results_by_type[qtype]["correct"].append(correct_flag)
        results_by_type[qtype]["total"] += 1

    # ---------- Save per-type results ----------
    for qtype, data in results_by_type.items():
        type_dir = os.path.join(args.out_dir, qtype)
        os.makedirs(type_dir, exist_ok=True)

        out_path = os.path.join(type_dir, f"{qtype}_results.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Question_ID", "Type", "Predicted", "Correct", "Match"])
            writer.writerows(data["rows"])

    # ---------- Save master results ----------
    master_csv_path = os.path.join(args.out_dir, "results.csv")
    with open(master_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question_ID", "Type", "Predicted", "Correct", "Match"])
        writer.writerows(all_results)

    print(f"\nMaster results saved to {master_csv_path}")

    # ---------- Summary ----------
    print("\n=== Accuracy by Category ===")
    for qtype, data in results_by_type.items():
        acc = "N/A"
        if data["correct"]:
            acc = sum(data["correct"]) / len(data["correct"])
        print(f"{qtype:16s}: {acc}")
    print("Done.")


if __name__ == "__main__":
    main()
