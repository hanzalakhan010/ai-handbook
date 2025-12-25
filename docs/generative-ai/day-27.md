---
sidebar_position: 28
id: day-27
title: 'Day 27: Fine-tuning LLMs with PEFT (LoRA)'
---

## Day 27: Fine-tuning LLMs with PEFT (LoRA)

### Objective

Understand the challenges of fine-tuning massive Large Language Models and learn about Parameter-Efficient Fine-Tuning (PEFT) methods, specifically LoRA, which allow for efficient and effective adaptation of LLMs to new tasks with minimal computational resources.

### Core Concepts

*   **The Problem with Full Fine-Tuning:**
    *   LLMs have billions, even hundreds of billions, of parameters.
    *   **Computational Cost:** Fine-tuning the *entire* model requires immense GPU memory and time.
    *   **Storage Cost:** Saving a fully fine-tuned version for each task means storing a full copy of the entire LLM, which is impractical.
    *   **Catastrophic Forgetting:** Fine-tuning all parameters on a small dataset can sometimes lead to the model "forgetting" its broad general knowledge learned during pre-training.

*   **Parameter-Efficient Fine-Tuning (PEFT): The Solution**
    *   The core idea is to fine-tune only a *small subset* of the model's parameters, or introduce a *small number of new parameters*, while keeping most of the original pre-trained weights frozen.
    *   This significantly reduces computational cost, memory requirements, and storage, while often achieving performance comparable to full fine-tuning.

*   **LoRA (Low-Rank Adaptation of Large Language Models):**
    *   One of the most popular and effective PEFT techniques.
    *   **The Big Idea:** Instead of directly fine-tuning the large weight matrices in a pre-trained Transformer model (e.g., in the attention layers), LoRA proposes to freeze these original weights and inject small, low-rank matrices into the Transformer layers.
    *   During fine-tuning, *only these newly added low-rank matrices* are trained. The original pre-trained weights remain fixed.

*   **How LoRA Works (Simplified):**
    1.  Consider a pre-trained weight matrix `W_0` (e.g., `10000x10000` parameters).
    2.  LoRA proposes to approximate the update to this matrix, `ΔW`, by a product of two smaller matrices: `ΔW = A · B`.
        *   `A`: A matrix of size `10000 x r` (where `r` is the "rank", a small number like 4, 8, or 16).
        *   `B`: A matrix of size `r x 10000`.
    3.  The number of parameters in `A` and `B` combined (`10000*r + r*10000`) is vastly smaller than the parameters in `ΔW` (`10000*10000`).
    4.  So, instead of training `ΔW`, you train `A` and `B`. When you want to use the fine-tuned model, you compute `W_0 + A·B`.

*   **Benefits of LoRA:**
    *   **Reduced Training Cost:** Only a tiny fraction of parameters are trained (e.g., 0.01% - 0.1% of the original model).
    *   **Reduced Memory Usage:** Less memory for gradients and optimizer states.
    *   **Faster Training:** Because fewer parameters need updating.
    *   **Efficient Deployment:** For a new task, you only need to store `A` and `B` (which are small), not a full copy of the base model. At inference, `A·B` is added to `W_0`.

### 🧠 Math & Stats Focus: Low-Rank Approximation

*   **Matrix Rank:** A fundamental concept in linear algebra. The rank of a matrix is the maximum number of linearly independent column vectors (or row vectors). A low-rank matrix means its columns (and rows) can be expressed as linear combinations of a small number of basis vectors.
*   **Singular Value Decomposition (SVD):** Any matrix `M` can be decomposed into `U · Σ · V^T`, where `Σ` contains the singular values. A low-rank approximation is achieved by keeping only the largest `r` singular values and their corresponding vectors.
*   **LoRA's Hypothesis:** The "update" `ΔW` that an LLM needs during fine-tuning to adapt to a new task is inherently a low-rank matrix. By enforcing this low-rank structure with matrices `A` and `B`, LoRA effectively captures the essence of the fine-tuning changes.

### 📜 Key Research Paper

*   **Paper:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
*   **Link:** [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
*   **Contribution:** This paper introduced LoRA, a game-changing PEFT method. It demonstrated that by adapting only a very small number of parameters, LoRA could achieve performance comparable to or even better than full fine-tuning, while drastically reducing computational and storage costs. This made fine-tuning LLMs accessible to a much wider range of researchers and practitioners.

### 💻 Project: Fine-tune a Small LLM with LoRA

Use the `peft` library from Hugging Face to fine-tune a small LLM (like GPT-2) with LoRA on a custom text generation task.

1.  **Install Libraries:** `pip install transformers peft datasets trl accelerate bitsandbytes`. (The last two are for efficient training).
2.  **Load a Dataset:** Use a small instruction dataset (e.g., the one you created on Day 11, or a pre-existing one from `datasets` like `tatsu-lab/alpaca_farm`).
3.  **Load a Base Model and Tokenizer:**
    *   `from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig`
    *   `model_id = "gpt2"`
    *   Load the model in 4-bit quantization (to save memory): `bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`
    *   `model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})`
    *   `tokenizer = AutoTokenizer.from_pretrained(model_id)`
    *   Set `tokenizer.pad_token = tokenizer.eos_token`
4.  **Configure LoRA:**
    *   `from peft import LoraConfig, get_peft_model`
    *   `lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["query_key_value"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")` (Here, `r` is the rank).
    *   `model = get_peft_model(model, lora_config)`
    *   `model.print_trainable_parameters()` (You should see a tiny fraction of the total parameters being trainable).
5.  **Fine-tune with `trl` (Transformer Reinforcement Learning):**
    *   The `trl` library provides an `SFTTrainer` that simplifies fine-tuning.
    *   Tokenize your dataset according to the model's tokenizer.
    *   Set up training arguments (e.g., `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`).
    *   `trainer = SFTTrainer(model, tokenizer, ...)`
    *   `trainer.train()`
6.  **Test the Fine-tuned Model:**
    *   Generate text with the fine-tuned LoRA model using a prompt from your dataset.
    *   Compare it to the base model's response. The LoRA model should now follow your custom instructions better.

### ✅ Progress Tracker

*   [ ] I can list at least 3 reasons why full fine-tuning of LLMs is challenging.
*   [ ] I can explain the core idea of Parameter-Efficient Fine-Tuning (PEFT).
*   [ ] I understand, at a high level, how LoRA injects low-rank matrices to adapt a model.
*   [ ] I have successfully fine-tuned a small LLM using LoRA and verified its new behavior.
