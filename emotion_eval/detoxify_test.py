
from detoxify import Detoxify

# each model takes in either a string or a list of strings

# results = Detoxify('original').predict('siktir git')["toxicity"]

# results = Detoxify('unbiased').predict(['siktir git'])["toxicity"]

results = Detoxify('multilingual').predict(["иди на хуй"])["toxicity"]

# # to specify the device the model will be allocated on (defaults to cpu), accepts any torch.device input

# model = Detoxify('original', device='cuda')

# optional to display results nicely (will need to pip install pandas)

import pandas as pd

print(results)
