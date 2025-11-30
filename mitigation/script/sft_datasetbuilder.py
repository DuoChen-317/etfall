import json

INPUT = "../../result/qwen/toxigen/text_results_Qwen_Qwen2.5-7B-Instruct_3000_toxigen.json"
OUTPUT = "sft_dataset.jsonl"         # final dataset


INSTRUCTION = "Please generate an appropriate and respectful response to the following text:\n\n"

def raw_clean():
    with open(INPUT, "r", encoding="utf-8") as fin, \
         open(OUTPUT, "w", encoding="utf-8") as fout:

        data = json.load(fin)

        for item in data:
            toxicity = item["toxicity"]
            if toxicity > 0.05:
                continue
            user_text = item["input"]
            teacher_output = item["output"]
            lang = item["lang"]

            user_message = f"Please generate an appropriate and respectful response to the following text in {lang}:\n\n{user_text}"

            sft_sample = {
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": teacher_output}
                ]
            }

            fout.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")

    print("Saved dataset:", OUTPUT)

INPUT_FILE = "../dataset/sft_dataset_from_gpt.jsonl"  # your original file
OUTPUT_FILE = "../dataset/sft_dataset.jsonl"  # cleaned file

def clean_record(rec):
    """
    Keep only fields needed for SFT:
    - 'input'/'output' OR
    - 'messages'
    Remove all metadata such as 'meta', 'source_id', 'toxicity', etc.
    """
    clean = {}

    # If using messages format
    if "messages" in rec:
        clean["messages"] = rec["messages"]

    # If using input/output format
    if "input" in rec and "output" in rec:
        clean["input"] = rec["input"]
        clean["output"] = rec["output"]

    return clean

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
            open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            cleaned = clean_record(rec)
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

    print(f"✔ Cleaning complete. Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()





