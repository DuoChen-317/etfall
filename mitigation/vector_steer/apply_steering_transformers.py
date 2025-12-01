import json
import numpy as np
from tqdm import tqdm
from detoxify import Detoxify
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================================================
# CONFIG
# ================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
HF_DATASET = "Tiyamo317/toxigen_multilinguish"
VECTOR_FILE = "safety_vector_en.npy"

OUTPUT_FILE = "steering_eval_results_en.jsonl"

ALPHA = 3.0
MAX_NEW_TOKENS = 128
DEVICE = "cuda"
LAYER = -1

# ================================================
# LOAD DATASET (HuggingFace)
# ================================================
print("Loading dataset from HuggingFace...")
ds = load_dataset(HF_DATASET, split="train")

# English column is ds["en"]
prompts = ds["en"]
ids = ds["id"]

print(f"Loaded {len(prompts)} English toxic prompts.")


# ================================================
# LOAD MODEL + VECTOR
# ================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)
model.eval()

steering_vec = torch.tensor(np.load(VECTOR_FILE)).to(DEVICE)
tox_model = Detoxify("multilingual")


# ================================================
# GENERATION FUNCTIONS
# ================================================
def get_tox(text):
    try:
        return tox_model.predict(text)["toxicity"]
    except:
        return 0.0


def generate_baseline(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def generate_with_steering(prompt, alpha=ALPHA):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Forward pass to get hidden states
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[LAYER]          # [1, seq, hidden]

    # Mean pooling
    h_mean = h.mean(dim=1)                    # [1, hidden]

    # Add vector
    steered_h = h_mean + alpha * steering_vec

    # Patch embedding of last token
    embeds = model.get_input_embeddings()(inputs["input_ids"])
    embeds[:, -1, :] = steered_h

    # Generate continuation
    out_tokens = model.generate(
        inputs_embeds=embeds,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )
    return tokenizer.decode(out_tokens[0], skip_special_tokens=True)


# ================================================
# RUN EVALUATION LOOP
# ================================================
print("\nRunning evaluation on 3000 English toxic prompts...\n")

results = []

for idx, prompt in tqdm(list(zip(ids, prompts))):
    baseline = generate_baseline(prompt)
    steered = generate_with_steering(prompt)

    tox_base = get_tox(baseline)
    tox_steer = get_tox(steered)

    record = {
        "id": int(idx),
        "prompt": prompt,
        "baseline": baseline,
        "steered": steered,
        "toxicity_baseline": tox_base,
        "toxicity_steered": tox_steer
    }
    results.append(record)

    # Save incremental
    with open(OUTPUT_FILE, "a", encoding="utf-8") as fw:
        fw.write(json.dumps(record, ensure_ascii=False) + "\n")


# ================================================
# SUMMARY
# ================================================
tox_base = np.array([r["toxicity_baseline"] for r in results])
tox_steer = np.array([r["toxicity_steered"] for r in results])

print("\n================ SUMMARY ================\n")
print(f"Mean baseline toxicity: {tox_base.mean():.4f}")
print(f"Mean steered toxicity:  {tox_steer.mean():.4f}")
print(f"Reduction:              {(tox_base.mean() - tox_steer.mean()):.4f}")
print(f"Improved samples:       {np.sum(tox_steer < tox_base)}/{len(results)}")
print("\n=========================================\n")
