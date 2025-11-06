import json

# read the bias result from json

import json

# ---- Step 1: Read the JSON file ----
with open("bias_results.json", "r") as f:
    data = json.load(f)

# ---- Step 2: Extract std_dev and range values ----
std_devs = [item["bias_values"]["std_dev"] for item in data]
ranges = [item["bias_values"]["range"] for item in data]

# ---- Step 3: Print or use the lists ----
print("Standard deviations:", std_devs)
print("Ranges:", ranges)
