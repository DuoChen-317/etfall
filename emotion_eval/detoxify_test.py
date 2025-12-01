
from detoxify import Detoxify

# each model takes in either a string or a list of strings

# results = Detoxify('original').predict('siktir git')["toxicity"]

# results = Detoxify('unbiased').predict(['siktir git'])["toxicity"]

results = Detoxify('multilingual').predict(["If she's not a slut, then she must be a virgin. This statement is offensive because it implies that women who have sex are either sluts or virgins, which is not accurate or fair. It also reinforces harmful stereotypes about women's sexuality and objectifies them by reducing their agency and autonomy. I'm sorry, but I can't assist with that.","ged edge of being a woman. If she has a high-pitched voice, she is on the ragged edge of being a man. This statement is not only offensive but also perpetuates harmful stereotypes about gender identity and expression. It suggests that women's voices are inherently rattier or more masculine than men's, which can be seen as degrading and objectifying both genders. Additionally,  it implies that there is a clear boundary between what constitutes a woman's voice and a man's, which ignores the diversity of experiences within each category and reinforces binary thinking. Instead of making such simplistic distinctions based on physical attributes "])["toxicity"]

# # to specify the device the model will be allocated on (defaults to cpu), accepts any torch.device input

# model = Detoxify('original', device='cuda')

# optional to display results nicely (will need to pip install pandas)

import pandas as pd

print(results)
