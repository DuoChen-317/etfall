# read the bias result from json
import json
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

FILE_PATHs = ["../result/qwen/base/text_results_Qwen_Qwen2.5-1.5B-Instruct_3000_base.json",
               "../result/qwen/base/text_results_Qwen_Qwen2.5-3B-Instruct_3000_base.json",
               "../result/qwen/base/text_results_Qwen_Qwen2.5-7B-Instruct_3000_base.json"]
MODEL_NAMES = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]

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


plot_number_toxicity_by_language(FILE_PATHs, MODEL_NAMES, output_path="model_comparison_counts_0.1.png", threshold=0.5)