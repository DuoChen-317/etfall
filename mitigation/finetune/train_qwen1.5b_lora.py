from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA = "qwen_sft_final.jsonl"

# Load dataset
dataset = load_dataset("json", data_files=DATA, split="train")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype="bfloat16"
)

# LoRA config (stronger on small models)
peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training config
training_args = SFTConfig(
    output_dir="./qwen_1_5b_sft_lora",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_seq_length=512,
    save_steps=500,
    logging_steps=50,
    warmup_ratio=0.1,
    bf16=True,
    fp16=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config
)

trainer.train()
