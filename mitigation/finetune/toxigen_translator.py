import json
import time
from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"

# NLLB language codes
SRC_LANG = "eng_Latn"
TARGET_LANGS = {
    "es": "spa_Latn",   # Spanish
    "fr": "fra_Latn",   # French
    "ru": "rus_Cyrl",   # Russian
    "tr": "tur_Latn",   # Turkish
}

MAX_SAMPLES = 3000
OUT_JSONL = "toxigen_multilingual_nllb.jsonl"
MIN_TOXICITY = 3


# -----------------------------
# Load NLLB model and tokenizer
# -----------------------------
def load_nllb_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NLLB model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME,use_safetensors=True)
    model.to(device)
    model.eval()
    return model, tokenizer, device


# -----------------------------
# Translation function
# -----------------------------
def translate_nllb(model, tokenizer, device, text, tgt_lang_code, max_new_tokens=128):
    # Set source language
    tokenizer.src_lang = SRC_LANG

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)




    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id("fr"),
            max_new_tokens=max_new_tokens,
            num_beams=4,
        )

    out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return out.strip()


# -----------------------------
# Build multilingual toxic dataset
# -----------------------------
def build_translated_dataset(
    max_samples=MAX_SAMPLES,
    out_jsonl=OUT_JSONL,
    min_toxicity=MIN_TOXICITY,
):
    print("Loading annotated ToxiGen (toxigen/toxigen-data, split=train)...")
    ds = load_dataset("toxigen/toxigen-data", split="train")

    # Filter only toxic samples
    ds = ds.filter(lambda x: x["toxicity_human"] >= min_toxicity)
    print(f"Filtered dataset size (toxicity_human >= {min_toxicity}): {len(ds)}")

    # Shuffle and select subset
    ds = ds.shuffle(seed=42)
    n = min(max_samples, len(ds))
    ds = ds.select(range(n))
    print(f"Using {n} samples for translation.")

    model, tokenizer, device = load_nllb_model()

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for i in tqdm(range(n), desc="Translating"):
            row = ds[i]
            en_text = row["text"]
            tox_level = row["toxicity_human"]

            record = {
                "text_en": en_text,
                "toxicity_human": tox_level,
                "translations": {},
            }

            # Translate into all target languages
            for code, lang_code in TARGET_LANGS.items():
                try:
                    translated = translate_nllb(
                        model, tokenizer, device, en_text, lang_code
                    )
                    record["translations"][code] = translated
                except Exception as e:
                    print(f"\n[ERROR] sample {i}, lang {code}: {e}")
                    time.sleep(1)
                    continue
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n🎉 Saved multilingual toxic dataset → {out_jsonl}")


# -----------------------------
if __name__ == "__main__":
    build_translated_dataset()
