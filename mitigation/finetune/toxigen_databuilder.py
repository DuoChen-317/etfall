import argparse
import json
from tqdm import tqdm
from detoxify import Detoxify
from vllm import LLM, SamplingParams

REWRITE_PROMPT = """Rewrite the following text in the SAME LANGUAGE({lang}).
Keep the meaning, but remove toxic, hateful, discriminatory, or harmful language.
Maintain a polite and neutral tone.

Text: {text}
"""

def build_prompt(text,lang):
    return REWRITE_PROMPT.format(text=text,lang=lang)

def build_sft_item(original, safe,original_toxicity,safe_toxicity):
    """Convert to SFT instruction format."""
    return {
        "instruction": "Rewrite the following text into a safe, non-toxic, polite version.",
        "input": original,
        "output": safe,
        "original_toxicity": original_toxicity,
        "safe_toxicity": safe_toxicity
    }

def score_toxicity(detox, text):
    """Return toxicity score from Detoxify."""
    if not text or text.strip() == "":
        return 0.0
    scores = detox.predict(text)
    return float(scores["toxicity"])

def process_batch(llm, sampling, detox, batch_prompts, batch_metadata, out_file):
    """Generate safe rewrites + toxicity scores for a batch."""
    results = llm.generate(batch_prompts, sampling)

    for (lang, original), output in zip(batch_metadata, results):
        safe = output.outputs[0].text.strip()

        # Toxicity scores
        original_tox = score_toxicity(detox, original)
        safe_tox = score_toxicity(detox, safe)

        item = build_sft_item(original, safe, original_tox, safe_tox)
        out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    out_file.flush()

def main(model_name, input_jsonl, output_jsonl, batch_size, max_tokens, gpu_mem):
    # load the llm model
    llm = LLM(model=model_name, max_model_len=2048, gpu_memory_utilization=gpu_mem)
    sampling = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=max_tokens)
    # load detoxify model
    detox = Detoxify('multilingual')


    out_file = open(output_jsonl, "w", encoding="utf-8")

    batch_prompts = []
    batch_metadata = []
    BATCH_SIZE = batch_size

    with open(input_jsonl, "r", encoding="utf-8") as f:
        # read input jsonl line by line
        for line in tqdm(f, desc="Processing lines"):
            data = json.loads(line)
            # get the toxic text
            toxic_en = data["text_en"]
            # get other language texts
            translations = data["translations"]

            # Add English
            batch_prompts.append(build_prompt(toxic_en,"en"))
            batch_metadata.append(("en", toxic_en))

            # Add other languages
            for lang in ["es", "fr", "ru", "tr"]:
                toxic_text = translations.get(lang, None)
                if toxic_text:
                    batch_prompts.append(build_prompt(toxic_text,lang))
                    batch_metadata.append((lang, toxic_text))

            # Run batch
            if len(batch_prompts) >= BATCH_SIZE:
                process_batch(llm, sampling, detox,
                              batch_prompts, batch_metadata, out_file)

                batch_prompts = []
                batch_metadata = []

    if batch_prompts:
        process_batch(llm, sampling, detox,
                      batch_prompts, batch_metadata, out_file)

    out_file.close()
    print("Finished. Output saved to:", output_jsonl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="translated.jsonl")
    parser.add_argument("--output", type=str, default="toxigen_multi_sft.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--gpu_mem", type=float, default=0.9)
    args = parser.parse_args()
    main(args.model, args.input, args.output, args.batch_size, args.max_tokens, args.gpu_mem)
