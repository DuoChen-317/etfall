import numpy as np

def bias_evaluation(toxicity_scores: list):
    """
    Evaluate bias for one sample based on toxicity scores across languages.

    Args:
        toxicity_scores (list): A list of toxicity scores for the same sample in different languages,
                                e.g. [0.42, 0.40, 0.38, 0.45, 0.43]

    Returns:
        dict: A dictionary containing bias metrics:
              - std_dev: standard deviation of scores across languages
              - range: max-min difference
              - mean: mean toxicity across languages
              - normalized_std: std_dev normalized by mean (coefficient of variation)
    """
    if not toxicity_scores:
        raise ValueError("toxicity_scores list is empty.")

    scores = np.array(toxicity_scores, dtype=float)
    mean_score = float(np.mean(scores))
    std_dev = float(np.std(scores, ddof=0))
    score_range = float(np.max(scores) - np.min(scores))
    normalized_std = float(std_dev / mean_score) if mean_score != 0 else np.nan

    return {
        "std_dev": std_dev,
        "range": score_range,
        "mean": mean_score,
        "normalized_std": normalized_std
    }