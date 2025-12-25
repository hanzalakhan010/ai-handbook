---
sidebar_position: 18
id: day-17
title: 'Day 17: Tokenization Deep Dive - The Building Blocks of LLMs'
---

## Day 17: Tokenization Deep Dive - The Building Blocks of LLMs

### Objective

Understand how modern language models process text through tokenization, learn about subword tokenization algorithms like BPE, and appreciate the trade-offs involved in choosing a vocabulary size.

### Core Concepts

*   **What is Tokenization?**
    *   The process of breaking down a piece of text into smaller units called "tokens".
    *   These tokens are the inputs that the model actually sees. Each token is mapped to a unique integer ID.

*   **Levels of Tokenization:**
    *   **Character-level:** Each character is a token.
        *   Pros: Small vocabulary, no "unknown" tokens.
        *   Cons: Sequences become very long, model has to learn word structure from scratch, computationally expensive.
    *   **Word-level:** Each word is a token, split by spaces and punctuation.
        *   Pros: Intuitive, sequences are shorter.
        *   Cons: Huge vocabulary required, suffers from "Out-of-Vocabulary" (OOV) problem (e.g., "tokenization" might not be in the vocab), doesn't handle morphology well (e.g., "run", "running", "ran" are three separate tokens).

*   **Subword Tokenization (The Modern Approach):**
    *   A hybrid approach that balances vocabulary size and sequence length.
    *   Common words are kept as single tokens (e.g., "the", "and").
    *   Rarer words are broken down into smaller, meaningful subword units.
    *   **Example:** "tokenization" -> `["token", "ization"]`. "unhappiness" -> `["un", "happi", "ness"]`.
    *   **Benefits:**
        *   Manages vocabulary size effectively.
        *   Essentially eliminates the OOV problem.
        *   Captures morphological information (e.g., the model can learn the meaning of the suffix "ness").

### 🧠 Math & Stats Focus: Byte-Pair Encoding (BPE)

BPE is a simple data compression algorithm that was adapted to become the most common subword tokenization method.

*   **The BPE Algorithm:**
    1.  **Initialize:** Start with a vocabulary consisting of all individual characters in your text corpus.
    2.  **Iterate:**
        a. Find the pair of adjacent tokens that occurs most frequently in the corpus.
        b. **Merge** this pair into a single new token.
        c. Add this new token to your vocabulary.
    3.  **Repeat:** Continue this process for a specified number of merges (the "vocabulary size" is a hyperparameter).

*   **Example Walkthrough:**
    1.  **Corpus:** `{"hug": 5, "pug": 2, "pun": 6, "bun": 4, "hugs": 5}`
    2.  **Initial Vocab:** `b, g, h, n, p, s, u`
    3.  **Step 1:** The most frequent pair is `u, g` (appears in hug, pug, hugs).
        *   **Merge:** `ug`.
        *   **New Vocab:** `b, g, h, n, p, s, u, ug`.
        *   **Corpus becomes:** `{"h ug": 5, "p ug": 2, "pun": 6, "bun": 4, "h ug s": 5}`
    4.  **Step 2:** The most frequent pair is now `u, n` (in pun, bun).
        *   **Merge:** `un`.
        *   **New Vocab:** `..., ug, un`.
        *   **Corpus becomes:** `{"h ug": 5, "p ug": 2, "p un": 6, "b un": 4, "h ug s": 5}`
    5.  ...and so on. The next merge would likely be `h, ug` to create `hug`.

### 📜 Key Research Paper

*   **Paper:** "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2015)
*   **Link:** [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)
*   **Contribution:** This paper introduced the use of the Byte-Pair Encoding (BPE) algorithm for subword tokenization in the context of neural machine translation. It showed that this method effectively handles rare and unknown words, significantly improving translation quality and becoming the standard for most subsequent Transformer-based models.

### 💻 Project: Use a Modern Tokenizer

Explore the tokenizer used by a real LLM like GPT-2 or BERT using the Hugging Face `transformers` library.

1.  **Load a Tokenizer:**
    *   `from transformers import AutoTokenizer`
    *   `tokenizer = AutoTokenizer.from_pretrained("gpt2")`
2.  **Inspect Vocabulary:**
    *   Check the vocabulary size: `tokenizer.vocab_size`.
    *   Look at some tokens: `tokenizer.get_vocab()`. Notice how it contains both full words and subword units.
3.  **Tokenize Sentences:**
    *   `text = "Tokenization is a fundamental step in natural language processing."`
    *   `tokens = tokenizer.tokenize(text)`
    *   `print(tokens)`
    *   Observe how the tokenizer breaks down the words. Which words are kept whole? Which are split? Notice how "Tokenization" is split into `['Token', 'ization']`.
4.  **Encode and Decode:**
    *   Encoding converts tokens into their integer IDs.
    *   `encoded = tokenizer.encode(text)`
    *   `print(encoded)`
    *   Decoding converts the IDs back into text.
    *   `decoded = tokenizer.decode(encoded)`
    *   `print(decoded)`

### ✅ Progress Tracker

*   [ ] I can describe the pros and cons of character-level, word-level, and subword-level tokenization.
*   [ ] I can explain the high-level steps of the BPE algorithm.
*   [ ] I have used a pre-trained tokenizer to see how it splits words into subword units.
*   [ ] I understand the difference between `tokenize`, `encode`, and `decode`.
