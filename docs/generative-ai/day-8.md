---
sidebar_position: 9
id: day-8
title: 'Day 8: The GPT Family and Causal Language Modeling'
---

## Day 8: The GPT Family and Causal Language Modeling

### Objective

Understand the architecture and training objective of the GPT (Generative Pre-trained Transformer) family of models, and contrast their "decoder-only" approach with BERT's "encoder-only" approach.

### Core Concepts

*   **GPT (Generative Pre-trained Transformer):**
    *   While BERT was designed for language *understanding* tasks (like classification), the GPT family was designed for language *generation*.
    *   GPT uses a **"decoder-only"** Transformer architecture. It's essentially just the decoder part of the original Transformer model, stacked on top of itself.

*   **Causal Language Modeling (CLM):**
    *   This is GPT's pre-training objective. It's a return to the classic language modeling task: **predict the next word**.
    *   `P(w_n | w_1, w_2, ..., w_{n-1})`
    *   It is "causal" because the prediction for a position can only depend on the words that came before it, not after.

*   **Masked Self-Attention:**
    *   How does a decoder-only Transformer, which can see the whole sequence at once, avoid "cheating" and looking at future words during training?
    *   It uses a special kind of self-attention called **masked self-attention**.
    *   Before the softmax is calculated in the self-attention step, a "mask" is applied that sets the attention scores for all future positions to negative infinity.
    *   After softmax, this means the model has a 0% probability of paying attention to any future word, enforcing the left-to-right, causal structure.

*   **BERT vs. GPT:**
    *   **BERT (Encoder-Only):**
        *   **Objective:** Masked Language Model (fills in blanks).
        *   **Sees:** Full bidirectional context.
        *   **Best for:** NLU tasks like sentiment analysis, question answering, feature extraction. It's great at producing rich representations of text.
    *   **GPT (Decoder-Only):**
        *   **Objective:** Causal Language Model (predicts the next word).
        *   **Sees:** Only left-to-right context.
        *   **Best for:** Natively generative tasks like text generation, summarization, chatbots, and zero-shot/few-shot prompting.

### 🧠 Math & Stats Focus: Autoregressive Generation

GPT models generate text autoregressively. This is a core concept in generative modeling.

*   **Autoregressive Model:** A model where the prediction for a time step `t` is regressed (dependent) on its own previous outputs.
    `x_t = f(x_{t-1}, x_{t-2}, ...)`
*   **The Generation Process (Decoding):**
    1.  You provide an initial prompt (e.g., "The cat sat on the").
    2.  The model processes this prompt and produces a probability distribution (via softmax) for the next word.
    3.  **Sampling:** You select a word from this distribution. There are several ways to do this:
        *   **Greedy Search:** Always pick the single most likely word. (Problem: Tends to be repetitive and boring).
        *   **Beam Search:** Keep track of the `k` most likely sequences at each step. Better than greedy, but still can be repetitive.
        *   **Top-k / Nucleus (Top-p) Sampling:** The most common methods for creative generation. They involve sampling randomly from a truncated portion of the probability distribution (e.g., only the top 50 most likely words, or words that make up the top 95% of the probability mass). This introduces randomness while avoiding nonsensical words.
    4.  The chosen word is appended to the input sequence.
    5.  The new, longer sequence is fed back into the model to predict the *next* word.
    6.  Repeat until a special `[END]` token is generated or a max length is reached.

### 📜 Key Research Paper

*   **Paper:** "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
*   **Link:** [https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
*   **Contribution:** This is the original GPT-1 paper. It demonstrated that a decoder-only Transformer, pre-trained on a causal language modeling objective and then fine-tuned, could achieve excellent performance on NLU tasks, rivaling the more complex encoder-decoder models of the time. It set the stage for GPT-2 and GPT-3.

### 💻 Project: Generate Text with GPT-2

Like BERT, the power of GPT models is in using pre-trained versions. Let's use Hugging Face to generate text with GPT-2, the first widely-released powerful generative model.

1.  **Install Hugging Face:** `pip install transformers`.
2.  **Load a Pre-trained GPT-2 and Tokenizer:**
    *   `from transformers import AutoTokenizer, AutoModelForCausalLM`
    *   `tokenizer = AutoTokenizer.from_pretrained('gpt2')`
    *   `model = AutoModelForCausalLM.from_pretrained('gpt2')`
3.  **Create a Text Generation Pipeline:** The easiest way to generate text is with a `pipeline`.
    *   `from transformers import pipeline`
    *   `generator = pipeline('text-generation', model='gpt2')`
4.  **Generate Text:** Call the generator with your prompt.
    *   `generator("In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains.", max_length=50, num_return_sequences=3)`
5.  **Experiment with Sampling:** Explore the arguments for the generator pipeline (or the `model.generate()` method), such as `do_sample=True`, `top_k=50`, and `top_p=0.95`. How does changing these parameters affect the creativity and coherence of the generated text?

### ✅ Progress Tracker

*   [ ] I can explain the difference between BERT's MLM objective and GPT's CLM objective.
*   [ ] I understand why GPT is a "decoder-only" architecture and how masked self-attention makes it work.
*   [ ] I can describe the autoregressive process of text generation.
*   [ ] I have used a pre-trained GPT-2 model to generate text.
