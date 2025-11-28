from vllm import LLM, SamplingParams
from detoxify import Detoxify
from collections import defaultdict
import os, sys, json
import argparse

# Import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.load_dataset import load_toxigen
from emotion_eval.emotional_eval import bias_evaluation


def main(output_dir: str, sample_number: int, prompt_template: str = "Respond to this text emotionally:",MODE: str = "base"):
    # ==== CONFIG ====
    DATASET = "Tiyamo317/toxigen_multilinguish"
    MODEL = "Tiyamo317/qwen2.5-1.5b-emotional"
    SUPPORTED_LANGUAGES = ["en", "fr", "ru", "es", "tr"]

    text_results = []
    bias_results = []

    # ==== LOAD DATA ====
    ds = load_toxigen(DATASET)

    llm = LLM(model=MODEL, max_model_len=2048, gpu_memory_utilization=0.9)
    sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
    detox = Detoxify('multilingual')

    # ==== GENERATION LOOP ====
    for lang in SUPPORTED_LANGUAGES:
        print(f"\n=== Processing language: {lang} ===")
        prompts = []
        for i in range(sample_number):
            sample_text = ds[i][lang]
            prompt = f"(language:{lang}, Do not switch to other languages.) {prompt_template}\n{sample_text}"
            prompts.append(prompt)

        print("Generating responses...")
        outputs = llm.generate(prompts, sampling)
        print("Generation completed!")

        print("Evaluating toxicity...")
        for i in range(sample_number):
            gen_text = outputs[i].outputs[0].text.strip()
            toxicity_score = float(detox.predict(gen_text)["toxicity"])
            text_results.append({
                "id": ds[i]["id"],
                "lang": lang,
                "input": ds[i][lang],
                "output": gen_text,
                "toxicity": toxicity_score
            })

    # ==== BIAS EVALUATION ====
    toxicity_by_id = defaultdict(list)
    for result in text_results:
        sample_id = result["id"]
        toxicity_by_id[sample_id].append(result["toxicity"])

    for sample_id, scores in toxicity_by_id.items():
        if len(scores) > 1:
            bias_values = bias_evaluation(scores)
            bias_results.append({"id": sample_id, "bias_values": bias_values})

    # ==== SAVE RESULTS ====
    model_savename = MODEL.replace('/','_')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"text_results_{model_savename}_{sample_number}_{MODE}.json"), "w", encoding="utf-8") as f:
        json.dump(text_results, f, indent=4, ensure_ascii=False)
    with open(os.path.join(output_dir, f"bias_results_{model_savename}_{sample_number}_{MODE}.json"), "w", encoding="utf-8") as f:
        json.dump(bias_results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Saved results in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate emotional bias of multilingual LLM.")
    parser.add_argument("--output_dir", type=str, default="../result/", help="Directory to save results.")
    parser.add_argument("--sample_number", type=int, default=500, help="Number of samples per language.")
    parser.add_argument("--prompt_template", type=str, default="Respond to this text emotionally:", help="Prompt template for text generation.")
    parser.add_argument("--mode", type=str, default="base")
    args = parser.parse_args()

    main(args.output_dir, args.sample_number, args.prompt_template, args.mode)
