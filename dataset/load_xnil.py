from datasets import load_dataset

ds = load_dataset("facebook/xnli", "all_languages")
print(ds)