import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_WEIGHTS = "./qwen_1_5b_sft_lora_final"
MERGED_OUTPUT = "./qwen2.5_1.5b_emotional_merged"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    model,
    LORA_WEIGHTS,
)

# Merge LoRA and unload adapter weights
model = model.merge_and_unload()

# Save merged full model (no LoRA needed anymore)
model.save_pretrained(MERGED_OUTPUT)
tokenizer.save_pretrained(MERGED_OUTPUT)

print("🎉 Merge complete! Saved to:", MERGED_OUTPUT)
