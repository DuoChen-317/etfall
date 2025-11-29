# read the bias result from json
import json
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

QWEN_BASE_FILE_PATHs = ["../result/qwen/base/text_results_Qwen_Qwen2.5-1.5B-Instruct_3000_base.json",
               "../result/qwen/base/text_results_Qwen_Qwen2.5-3B-Instruct_3000_base.json",
               "../result/qwen/base/text_results_Qwen_Qwen2.5-7B-Instruct_3000_base.json"]

QWEN_ADVERSARIAL_FILE_PATHs = ["../result/qwen/adversarial/text_results_Qwen_Qwen2.5-1.5B-Instruct_3000_adversarial.json",
               "../result/qwen/adversarial/text_results_Qwen_Qwen2.5-3B-Instruct_3000_adversarial.json",
               "../result/qwen/adversarial/text_results_Qwen_Qwen2.5-7B-Instruct_3000_adversarial.json"]

QWEN_SAFE_FILE_PATHs = ["../result/qwen/safe/text_results_Qwen_Qwen2.5-1.5B-Instruct_3000_safe.json",
               "../result/qwen/safe/text_results_Qwen_Qwen2.5-3B-Instruct_3000_safe.json",
               "../result/qwen/safe/text_results_Qwen_Qwen2.5-7B-Instruct_3000_safe.json"]

LLAMA_BASE_FILE_PATHs = ["../result/llama/base/text_results_meta-llama_Llama-3.1-8B-Instruct_3000_base.json",
               "../result/llama/base/text_results_meta-llama_Llama-3.2-3B-Instruct_3000_base.json",
               "../result/llama/base/text_results_meta-llama_Llama-3.1-8B-Instruct_3000_base.json"]

LLAMA_ADVERSARIAL_FILE_PATHs = ["../result/llama/adversarial/text_results_meta-llama_Llama-3.1-8B-Instruct_3000_adversarial.json",  
               "../result/llama/adversarial/text_results_meta-llama_Llama-3.2-3B-Instruct_3000_adversarial.json",
               "../result/llama/adversarial/text_results_meta-llama_Llama-3.1-8B-Instruct_3000_adversarial.json"]

LLAMA_SAFE_FILE_PATHs = ["../result/llama/safe/text_results_meta-llama_Llama-3.1-8B-Instruct_3000_safe.json",  
               "../result/llama/safe/text_results_meta-llama_Llama-3.2-3B-Instruct_3000_safe.json",
               "../result/llama/safe/text_results_meta-llama_Llama-3.1-8B-Instruct_3000_safe.json"]

QWEN_TOXIGEN_PATHS = ["../result/qwen/toxigen/text_results_Qwen_Qwen2.5-1.5B-Instruct_3000_toxigen.json",
            "../result/qwen/toxigen/text_results_Qwen_Qwen2.5-3B-Instruct_3000_toxigen.json",
            "../result/qwen/toxigen/text_results_Qwen_Qwen2.5-7B-Instruct_3000_toxigen.json",
            "../result/qwen/toxigen/text_results_Tiyamo317_qwen2.5-1.5b-emotional_3000_toxigen.json"]

MODEL_NAMES_QWEN = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]
MODEL_NAMES_LLAMA = ["Llama3.2-1B", "Llama3.2-3B", "Llama3.1-8B"]
MODEL_NAMES_QWEN_TOXIGEN = ["Qwen2.5-1.5B","Qwen2.5-3B", "Qwen2.5-7B","Qwen2.5-1.5B-FT"]

def plot_mean_toxicity_by_language(model_files, model_names, output_path="model_comparison.png",threshold=0.0):
    """
    model_files: list of JSON file paths for different models
    model_names: list of model names (same order as model_files)
    """

    # Check input lengths
    assert len(model_files) == len(model_names), "model_files and model_names must match in length"

    # ---- build structure: model -> lang -> list of toxicities ----
    scores = {name: defaultdict(list) for name in model_names}

    for file_path, model_name in zip(model_files, model_names):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            lang = item["lang"]
            toxicity = item["toxicity"]
            if toxicity >= threshold:
                scores[model_name][lang].append(toxicity)

    # ---- compute means ----
    # Collect all languages
    languages = sorted({lang for model in scores.values() for lang in model.keys()})

    # Matrix of means: rows = languages, columns = models
    mean_matrix = []
    for lang in languages:
        row = []
        for model_name in model_names:
            values = scores[model_name].get(lang, [])
            mean_val = sum(values) / len(values) if values else 0
            row.append(mean_val)
        mean_matrix.append(row)

    mean_matrix = np.array(mean_matrix)  # shape: (num_lang, num_models)

    # ---- Step 3: plot grouped bar chart ----
    num_lang = len(languages)
    num_models = len(model_names)

    x = np.arange(num_lang)  # language positions
    width = 0.8 / num_models # bar width per model

    plt.figure(figsize=(15, 7))

    for i, model_name in enumerate(model_names):
        plt.bar(x + i * width,
                mean_matrix[:, i],
                width=width,
                label=model_name)

    # ---- aesthetics ----
    plt.xticks(x + width * (num_models - 1) / 2, languages, rotation=45, ha="right")
    plt.ylabel("Mean Toxicity")
    plt.title("Mean Toxicity by Language Across Models")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=1200)
    print(f"[Saved] {output_path}")

def plot_number_toxicity_by_language(model_files, model_names, output_path="model_comparison_counts.png",threshold=0.0):
    """
    model_files: list of JSON file paths for different models
    model_names: list of model names (same order as model_files)
    """
    # ---- build structure: model -> lang -> list of toxicities ----
    scores = {name: defaultdict(list) for name in model_names}

    for file_path, model_name in zip(model_files, model_names):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            lang = item["lang"]
            toxicity = item["toxicity"]
            if toxicity >= threshold:
                scores[model_name][lang].append(toxicity)

    # ---- compute counts ----
    # Collect all languages
    languages = sorted({lang for model in scores.values() for lang in model.keys()})

    # Matrix of counts: rows = languages, columns = models
    count_matrix = []
    for lang in languages:
        row = []
        for model_name in model_names:
            values = scores[model_name].get(lang, [])
            count_val = len(values)
            row.append(count_val)
        count_matrix.append(row)

    count_matrix = np.array(count_matrix)  # shape: (num_lang, num_models)

    # ---- Step 3: plot grouped bar chart ----
    num_lang = len(languages)
    num_models = len(model_names)

    x = np.arange(num_lang)  # language positions
    width = 0.8 / num_models # bar width per models

    plt.figure(figsize=(15, 7))

    for i, model_name in enumerate(model_names):
        plt.bar(x + i * width,
                count_matrix[:, i],
                width=width,
                label=model_name)

    # ---- aesthetics ----
    plt.xticks(x + width * (num_models - 1) / 2, languages, rotation=45, ha="right")
    plt.ylabel("Number of Toxic Outputs")
    plt.title(f"Number of Toxic Outputs by Language Across Models (Threshold: {threshold})")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=1200)
    print(f"[Saved] {output_path}")


plot_mean_toxicity_by_language(QWEN_TOXIGEN_PATHS, MODEL_NAMES_QWEN_TOXIGEN, output_path="qwen_toxigen_mean.png",threshold=0.0)
plot_number_toxicity_by_language(QWEN_TOXIGEN_PATHS, MODEL_NAMES_QWEN_TOXIGEN, output_path="qwen_toxigen_number.png",threshold=0.5)