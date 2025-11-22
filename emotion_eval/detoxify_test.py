
from detoxify import Detoxify

# each model takes in either a string or a list of strings

# results = Detoxify('original').predict('siktir git')["toxicity"]

# results = Detoxify('unbiased').predict(['siktir git'])["toxicity"]



# # to specify the device the model will be allocated on (defaults to cpu), accepts any torch.device input

# model = Detoxify('original', device='cuda')

# optional to display results nicely (will need to pip install pandas)

import pandas as pd
input = "Some people in Argentina may hold traditional or sexist attitudes, but this does not apply to all Argentine men"
results = Detoxify('multilingual').predict([input])["toxicity"]
print(f"Input: {input}")
print("Toxicity:", results)
