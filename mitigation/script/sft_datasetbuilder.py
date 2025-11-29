import json

INPUT = "../../result/qwen/toxigen/text_results_Tiyamo317_qwen2.5-1.5b-emotional_3000_toxigen.json"       # your file
OUTPUT = "sft_dataset.jsonl"         # final dataset


INSTRUCTION = "Please generate an appropriate and respectful response to the following text:\n\n"

with open(INPUT, "r", encoding="utf-8") as fin, \
     open(OUTPUT, "w", encoding="utf-8") as fout:

    data = json.load(fin)

    for item in data:
        user_text = item["input"]
        teacher_output = item["output"]
        lang = item["lang"]

        user_message = f"(language:{lang}, Do not switch to other languages.)Please generate an appropriate and respectful response to the following text:\n\n{user_text}"

        sft_sample = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": teacher_output}
            ]
        }

        fout.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")

print("Saved dataset:", OUTPUT)

