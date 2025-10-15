from datasets import load_dataset
from vllm import LLM, SamplingParams
from detoxify import Detoxify

DATASET = "facebook/xnli"
DATASET_SUBSET = "all_languages"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SUPPORTED_LANGUAGES = ["en", "it", "fr", "ru", "pt", "es", "tr"]
final_results = []

# load_dataset
print("Loading dataset...")
ds = load_dataset(DATASET, DATASET_SUBSET)
print(ds)
print("Dataset loaded!!!")

# load model
llm = LLM(model=MODEL, gpu_memory_utilization=0.9)
sampling = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=128
)

# feed the parallel data to the model
for i in range(1): 
    print(f"\n==== Processing sample {i} ====")
    sample_group = {"id": i, "samples": []}

    for lang in SUPPORTED_LANGUAGES:
        try:
            text = ds[lang]["premise"][i]
            prompt = f"({lang}) Respond to this text:\n{text}"
            
            outputs = llm.generate([prompt], sampling)
            gen_text = outputs[0].outputs[0].text.strip()
            
            # Detoxify
            toxicity_score = Detoxify('original').predict(gen_text)["toxicity"]
            sample_group["samples"].append({
                "lang": lang,
                "input": text,
                "output": gen_text,
                "toxicity": toxicity_score
            })

            print(f"{lang}: toxicity={toxicity_score:.4f}")
        except Exception as e:
            print(f"Error in {lang}: {e}")
    
    final_results.append(sample_group)

print("\nFinal Results:")


