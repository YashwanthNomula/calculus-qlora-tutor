# QLoRA Fine-Tuning Case Study (Calculus Reasoning)

This repository contains a practical **LLM fine-tuning case study using QLoRA**,
focused on adapting an open-weight model for **structured, step-by-step reasoning**
under a **single-GPU, limited-VRAM local setup**.

A calculus-style reasoning task is used as a **controlled evaluation domain** to
study instruction-following behavior and output structure.

---

## What this project does

- Generates **synthetic calculus instruction–response data** (Alpaca-style format)
- Applies **QLoRA (4-bit quantization + LoRA adapters)** for parameter-efficient fine-tuning
- Iterates on **prompt patterns and data cleaning** to improve reasoning consistency
- Runs **base vs fine-tuned qualitative evaluation**
- Publishes the trained LoRA adapter to Hugging Face

---

## Key Technologies

- PyTorch
- Hugging Face Transformers
- PEFT / QLoRA
- bitsandbytes
- Accelerate

---

## Evaluation

To evaluate the impact of fine-tuning, a held-out test set of calculus-style
instruction prompts was used to compare the base model against the QLoRA
fine-tuned model.

Evaluation was qualitative and focused on instruction adherence and reasoning
structure.

Compared to the base model, the fine-tuned model:
- followed the Instruction → Response format more consistently
- produced clearer step-by-step explanations
- reduced repetition and malformed outputs
- showed more controlled and concise reasoning

The base model often produced correct answers but exhibited formatting drift,
repeated response blocks, and inconsistent output structure.

All evaluations were run on a **single-GPU, limited-VRAM local setup**, which
required CPU offloading and resulted in long inference times.

---

## Repository Contents

- `scripts/` – synthetic data generation, cleaning, training, and inference scripts
- `.gitignore` – excludes datasets and model artifacts

---

The trained LoRA adapter is published separately on Hugging Face.
	