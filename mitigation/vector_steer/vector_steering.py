import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PAIR_FILE = "../dataset/steering_pairs_filtered.jsonl"
OUTPUT_VECTOR = "safety_vector.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = -1    # use last hidden layer
EMB_POOLING = "mean"   # mean pooling

LANG = "en"


# ----------------------------
# LOAD MODEL
# ----------------------------
print("Loading model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()


# ----------------------------
# HELPER: GET EMBEDDING
# ----------------------------
def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True
        )
        hidden = outputs.hidden_states[LAYER]   # [1, seq, hidden]

    if EMB_POOLING == "mean":
        emb = hidden.mean(dim=1).squeeze(0)     # [hidden]
    else:
        emb = hidden[:, -1, :].squeeze(0)       # CLS-like

    return emb.float().cpu()


# ----------------------------
# LOAD PAIRS
# ----------------------------
print("Loading steering pairs...")
pairs = []
with open(PAIR_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item["lang"] == LANG:
            pairs.append(item)

print("Total steering pairs:", len(pairs))


# ----------------------------
# COMPUTE EMBEDDINGS
# ----------------------------
pos_embs = []
neg_embs = []

print("Encoding texts...")
for item in tqdm(pairs):
    pos_embs.append(get_embedding(item["positive"]))
    neg_embs.append(get_embedding(item["negative"]))

pos_embs = torch.stack(pos_embs)
neg_embs = torch.stack(neg_embs)

print("Positive shape:", pos_embs.shape)
print("Negative shape:", neg_embs.shape)


# ----------------------------
# COMPUTE STEERING VECTOR
# ----------------------------
mean_pos = pos_embs.mean(dim=0)
mean_neg = neg_embs.mean(dim=0)

steering = mean_pos - mean_neg
steering = steering / steering.norm()  # normalize

print("Steering vector shape:", steering.shape)


# ----------------------------
# SAVE
# ----------------------------
np.save(OUTPUT_VECTOR, steering.numpy())
print("Saved steering vector →", OUTPUT_VECTOR)
