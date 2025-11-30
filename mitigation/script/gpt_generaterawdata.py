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

OUTPUT_FILE = "../dataset/sft_dataset_from_gpt.jsonl"
MODEL_NAME = "gpt-4.1-mini"
MAX_SAMPLES = None          # set to e.g. 200 for testing
SLEEP_TIME = 0.5            # avoid rate limits

client = OpenAI()

BASE_INSTRUCTION = (
    "Please generate a safe, respectful, non-toxic response to the following text.\n"
    "Do not repeat harmful words.\n"
    "Reply in the same language.\n\n"
)
# load detoxify
dex = Detoxify("multilingual")
LANG_KEYS = ["en", "es", "fr", "ru", "tr"]


# --------------------------------------------
# LOAD DATASET
# --------------------------------------------
print(f"Loading dataset: {DATASET_NAME} ...")
dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
print(f"Loaded {len(dataset)} rows.")


# --------------------------------------------
# GENERATE
# --------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for idx, item in enumerate(dataset):

        if MAX_SAMPLES and idx >= MAX_SAMPLES:
            break

        # iterate through each language text
        for lang in LANG_KEYS:

            input_text = item[lang]

            if not input_text or len(input_text.strip()) == 0:
                continue

            user_prompt = f"{BASE_INSTRUCTION}{input_text}"

            try:
                response = client.responses.create(
                    model=MODEL_NAME,
                    input=user_prompt
                )
                safe_output = response.output_text.strip()

            except Exception as e:
                print(f"Error at sample {idx}, lang {lang}: {e}")
                continue

            toxicity = dex.predict(safe_output)["toxicity"]
            toxicity = float(toxicity)

            sft_item = {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": safe_output}
                ],
                "meta": {
                    "lang": lang,
                    "source_id": item["id"],
                    "toxicity_placeholder": toxicity
                }
            }

            fout.write(json.dumps(sft_item, ensure_ascii=False) + "\n")
            time.sleep(SLEEP_TIME)

print("✅ DONE — Saved SFT dataset to:", OUTPUT_FILE)
