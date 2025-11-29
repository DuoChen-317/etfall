from datasets import load_dataset


def load_idassigned_dataset(dataset = "facebook/xnli", subset = "all_languages"):
    # load_dataset
    print("Loading dataset...")
    ds = load_dataset(dataset, subset)
    ds = ds["train"]
    print("Dataset loaded!!!")
    ds = ds.add_column("id", list(range(len(ds))))
    lang = "ar"
    return ds

def load_toxigen(dataset = "Tiyamo317/toxigen_multilinguish", split="train"):
    # load_dataset
    print("Loading dataset...")
    ds = load_dataset(dataset,split=split)
    print("Dataset loaded!!!")
    return ds
