import json

INPUT = "../../result/qwen/toxigen/text_results_Qwen_Qwen2.5-7B-Instruct_3000_toxigen.json"
OUTPUT = "sft_dataset.jsonl"         # final dataset


INSTRUCTION = "Please generate an appropriate and respectful response to the following text:\n\n"

with open(INPUT, "r", encoding="utf-8") as fin, \
     open(OUTPUT, "w", encoding="utf-8") as fout:

    data = json.load(fin)

    for item in data:
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

