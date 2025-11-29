import json
import time
from datasets import load_dataset
from openai import OpenAI
from detoxify import Detoxify

# --------------------------------------------
# CONFIG
# --------------------------------------------
DATASET_NAME = "Tiyamo317/toxigen_multilinguish"
DATASET_SPLIT = "train"

OUTPUT_FILE = "sft_dataset_from_gpt.jsonl"
MODEL_NAME = "gpt-4.1-mini"   # or gpt-5-nano

MAX_SAMPLES = None
SLEEP_TIME = 0.4               # avoid rate limits

client = OpenAI()
detox = Detoxify("multilingual")

BASE_INSTRUCTION = (
    "Please generate a safe, respectful, non-toxic response to the following text.\n"
    "Avoid repeating slurs or harmful content. Respond in the same language.\n\n"
)

# --------------------------------------------
# LOAD HF DATASET
# --------------------------------------------
print(f"Loading dataset: {DATASET_NAME} ...")
dataset = load_dataset(DATASET_NAME, DATASET_SPLIT)
print(f"Loaded {len(dataset)} samples.")


# --------------------------------------------
# GENERATE SFT DATASET
# --------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for idx, item in enumerate(dataset):

        if MAX_SAMPLES and idx >= MAX_SAMPLES:
            break

        lang = item["lang"]
        input_text = item["premise"][lang]

        user_prompt = f"{BASE_INSTRUCTION}{input_text}"

        # Call GPT
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=user_prompt
            )
            assistant_output = response.output_text.strip()

            toxicity = float(detox.predict(assistant_output)["toxicity"])
        except Exception as e:
            print(f"❌ Error at sample {idx}: {e}")
            continue

        # Format SFT sample
        sft_sample = {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_output}
            ],
            "meta": {
                "lang": lang,
                "toxicity": toxicity
            }
        }

        fout.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")
        time.sleep(SLEEP_TIME)

print("✅ DONE — Saved SFT dataset to:", OUTPUT_FILE)
