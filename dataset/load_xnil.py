from datasets import load_dataset

ds = load_dataset("facebook/xnli", "all_languages")
ds = ds["train"]
text = ds["premise"][0]["en"]
print(text)
