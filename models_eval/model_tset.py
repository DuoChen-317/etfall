import os
import yaml
from huggingface_hub import InferenceClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CONFIG_PATH = "../config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    print("Configuration loaded:", config)

client = InferenceClient(
    provider="novita",
    api_key=config["hf_token"],
)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3.2-Exp",
    messages=[
        {
            "role": "user",
            "content": "Hi, AM I TALKING TO YOU? And just tell me your model name."
        }
    ],
)

completion_text = completion.choices[0].message.content
print("Model response:", completion_text)

analyzer = SentimentIntensityAnalyzer()
vs = analyzer.polarity_scores(completion_text)
print("{:-<65} {}".format(completion_text, str(vs)))