import json
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
N_INPUT_JSON = "text_results_Qwen_Qwen2.5-1.5B-Instruct_3000_toxigen.json"
P_INPUT_JSONL = "../dataset/sft_dataset_from_gpt.jsonl"
OUTPUT_JSONL = "steering_pairs_filtered.jsonl"

TOXICITY_THRESHOLD = 0.5

# ----------------------------
# LOAD QWEN NEGATIVE DATA
# ----------------------------
print("Loading Qwen toxic outputs...")
with open(N_INPUT_JSON, "r", encoding="utf-8") as f:
    qwen_data = json.load(f)

# Index like: qwen_by[(id, lang)]
qwen_by = {}
for item in qwen_data:
    key = (item["id"], item["lang"])
    qwen_by[key] = item

# ----------------------------
# LOAD GPT SAFE DATA
# ----------------------------
print("Loading GPT safe outputs...")
gpt_by = {}

with open(P_INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        sid = item["meta"]["source_id"]  # integer ID
        lang = item["meta"]["lang"]  # language code

        # key is (id, lang)
        key = (sid, lang)

        gpt_by[key] = item

# ----------------------------
# BUILD MULTILINGUAL PAIRS
# ----------------------------
print("Building matched multilingual pairs...")

pairs = []

for key, qwen_item in tqdm(qwen_by.items()):
    qid, lang = key

    if qwen_item["toxicity"] < TOXICITY_THRESHOLD:
        continue

    if key not in gpt_by:
        # no safe pair for this language
        continue

    gpt_item = gpt_by[key]

    # GPT messages
    safe_reply = gpt_item["messages"][1]["content"]

    pairs.append({
        "id": qid,
        "lang": lang,
        "prompt": qwen_item["input"],
        "negative": qwen_item["output"],
        "positive": safe_reply,
        "toxicity_score": qwen_item["toxicity"]
    })

print("Total multilingual high-toxicity pairs:", len(pairs))

# ----------------------------
# SAVE OUTPUT JSONL
# ----------------------------
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print("Saved:", OUTPUT_JSONL)
