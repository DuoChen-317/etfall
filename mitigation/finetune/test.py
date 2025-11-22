from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# 1. Choose a pre-trained NLLB model
# You can choose different sizes, e.g., "facebook/nllb-200-distilled-600M" or "facebook/nllb-200-1.3B"
model_name = "facebook/nllb-200-distilled-600M"

# 2. Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. Define source and target languages using BCP-47 codes
# For example, English (Latin script) to French (Latin script)
source_lang = "eng_Latn"
target_lang = "fra_Latn"

# 4. Create a translation pipeline
translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=source_lang, tgt_lang=target_lang)

# 5. Provide the text to be translated
text_to_translate = "Hello, how are you doing today?"

# 6. Perform the translation
output = translator(text_to_translate, max_length=400) # max_length can be adjusted
translated_text = output[0]["translation_text"]

# 7. Print the translated text
print(f"Original text: {text_to_translate}")
print(f"Translated text ({target_lang}): {translated_text}")