# model_training-supervised_fine_tuning

# 🦙 Alpaca + GPT‑Neo‑125M Fine‑Tuning

This project demonstrates **supervised fine‑tuning (SFT)** of the Hugging Face model **EleutherAI/gpt‑neo‑125M** on the **Alpaca dataset**.  
The goal is to adapt a base language model to follow instructions more effectively by training it on `(instruction, input, output)` pairs.

---

## 📌 Project Overview
- **Base Model**: `EleutherAI/gpt-neo-125M` (causal language model, lightweight and fast).
- **Dataset**: [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (52k instruction‑response pairs).
- **Frameworks**: Hugging Face `transformers`, `datasets`, `accelerate`.
- **Training Objective**: Next‑token prediction on formatted instruction‑response text.
- **Hardware**: Runs on GPU (CUDA/MPS) or TPU (via PyTorch/XLA).

---

## ⚙️ Setup

Install dependencies:
```bash
pip install -U transformers datasets accelerate sentencepiece huggingface_hub

📂 Data Preparation
1. Load the Alpaca dataset:

from datasets import load_dataset
dataset = load_dataset("tatsu-lab/alpaca")

2. Format examples into a single text string:

def format_example(example):
    if example.get("input"):
        prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput:"
    else:
        prompt = f"Instruction: {example['instruction']}\nOutput:"
    return {"text": prompt + example["output"]}

3. Tokenize with GPT‑Neo tokenizer:

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue

🏋️ Training
Use Hugging Face Trainer:

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
    warmup_steps=100,
    weight_decay=0.01,
    bf16=True,   # ✅ use bf16 on TPU, fp16 on CUDA
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset.select(range(1000)),  # small eval subset
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

📊 Evaluation
After training, test inference:

from transformers import pipeline

pipe = pipeline("text-generation", model="./results", tokenizer=tokenizer)
print(pipe("Instruction: Write a poem about F1 racing\nOutput:", max_new_tokens=100)[0]["generated_text"])

🖥️ Hardware Notes
• Mac (MPS): Use small batch sizes (2–4). Models >1.3B params may OOM.
• GPU (CUDA): Enable fp16=True for faster training.
• TPU (v5e‑8): Use bf16=True and optim="adamw_torch_xla" if supported. Large batch sizes (16–32) are possible.

---
🚀 Key Learnings
• GPT‑Neo doesn’t define a pad token → must set tokenizer.pad_token = tokenizer.eos_token.
• On TPU, avoid fused optimizers → use adamw_torch_xla.
• Hugging Face Trainer simplifies fine‑tuning but requires version‑compatible arguments.

📌 Next Steps
• Try larger models (gpt-neo-1.3B, opt-350m) if hardware allows.
• Experiment with quantization (8‑bit/4‑bit) for memory efficiency.
• Evaluate outputs with BLEU/ROUGE or human preference scoring.
