import json

# read the bias result from json

FILE_PATHs = ["../result/bias_results_Qwen_Qwen2.5-1.5B-Instruct_3000_base.json",
               "../result/bias_results_Qwen_Qwen2.5-3B-Instruct_3000_base.json",
               "../result/bias_results_Qwen_Qwen2.5-7B-Instruct_3000_base.json"]

def calculate_mean_std_dev_and_range(file_path):
    # ---- Step 1: Read the JSON file ----
    with open(file_path, "r") as f:
        data = json.load(f)

    # ---- Step 2: Extract std_dev and range values ----
    std_devs = [item["bias_values"]["std_dev"] for item in data]
    log_dev = [item["bias_values"]["log_std"] for item in data]
    normolized_std = [item["bias_values"]["normalized_std"] for item in data]
    ranges = [item["bias_values"]["range"] for item in data]

    # calculate mean std_dev
    mean_std_devs = sum(std_devs) / len(std_devs)
    mean_ranges = sum(ranges) / len(ranges)
    mean_log_dev = sum(log_dev) / len(log_dev)
    mean_normalized_std = sum(normolized_std) / len(normolized_std)

    # ---- Step 3: Print or use the lists ----
    print("mean_std_dev =", mean_std_devs)
    print("mean_range =", mean_ranges)
    print("mean_log_dev =", mean_log_dev)
    print("mean_normalized_std =", mean_normalized_std)

for FILE_PATH in FILE_PATHs:
    print(f"Results for {FILE_PATH}:")
    calculate_mean_std_dev_and_range(FILE_PATH)
    print("\n")