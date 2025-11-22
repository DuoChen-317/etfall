from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA = "qwen_sft_final.jsonl"

# Load dataset
dataset = load_dataset("json", data_files=DATA, split="train")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto"
)

# LoRA config (passed directly to trainer)
peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Formatting function: converts messages → chat text
def format_messages(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return text

# Training config
training_args = SFTConfig(
    output_dir="./qwen_1_5b_sft_lora",
    max_length=512,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=50,
    save_steps=500,
    bf16=True,
    report_to="none",
)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,    # <-- REQUIRED
    peft_config=peft_config,       # <-- TRL 0.25 supports it
    formatting_func=format_messages,  # <-- REQUIRED
)

trainer.train()
