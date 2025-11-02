from datasets import load_dataset


def load_idassigned_dataset(dataset = "facebook/xnli", subset = "all_languages"):
    # load_dataset
    print("Loading dataset...")
    ds = load_dataset(dataset, subset)
    ds = ds["train"]
    print("Dataset loaded!!!")
    ds = ds.add_column("id", list(range(len(ds))))
    lang = "ar"
    print(ds[0]["premise"][lang])
    return ds


