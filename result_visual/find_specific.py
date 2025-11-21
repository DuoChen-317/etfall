import json
from collections import defaultdict

# find the toxify that larger than threshold
def find_specific_toxicity(input_file: str, output_file: str, threshold: float = 0.5):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_results = []
    for item in data:
        if item["toxicity"] >= threshold:
            filtered_results.append(item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_results, f, ensure_ascii=False, indent=4)

    print(f"[Saved] Filtered results with toxicity >= {threshold} to {output_file}")

# Example usage
input_file_path = "../result/qwen/safe/text_results_Qwen_Qwen2.5-7B-Instruct_3000_safe.json"
output_file_path = "../figure/qwen/qwen_safe_specific_threshold_0.5.json"
toxicity_threshold = 0.5

find_specific_toxicity(input_file_path, output_file_path, toxicity_threshold)