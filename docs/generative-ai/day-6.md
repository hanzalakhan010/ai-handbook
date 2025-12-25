---
sidebar_position: 7
id: day-6
title: 'Day 6: The Transformer Architecture'
---

## Day 6: The Transformer Architecture - "Attention Is All You Need"

### Objective

Understand the architecture of the Transformer model, which abandoned recurrence and convolutions entirely in favor of self-attention, setting the stage for all modern LLMs.

### Core Concepts

*   **The Problem with Recurrence (RNNs/LSTMs):**
    *   **Sequential Computation:** RNNs process text word by word, which is inherently slow and cannot be parallelized.
    *   **Long-Range Dependencies:** While attention helped, passing information over very long distances was still challenging for RNNs.

*   **The Transformer's Big Idea:**
    *   What if we get rid of the sequential recurrence entirely?
    *   What if we use attention to relate every word in the input directly to every other word? This is called **Self-Attention**.

*   **Transformer Architecture (High-Level):**
    *   It's still an **Encoder-Decoder** structure, but the internal components are new.
    *   **The Encoder Stack:** A stack of identical "Encoder Layers". Each layer has two sub-layers:
        1.  A **Multi-Head Self-Attention** mechanism.
        2.  A simple, position-wise **Feed-Forward Neural Network**.
    *   **The Decoder Stack:** A stack of "Decoder Layers". Each layer has three sub-layers:
        1.  A **Masked Multi-Head Self-Attention** mechanism (to prevent it from "cheating" by looking at future words during training).
        2.  A **Multi-Head Cross-Attention** mechanism (this is where it pays attention to the encoder's output).
        3.  A position-wise **Feed-Forward Neural Network**.

*   **Positional Encodings:**
    *   Since there's no recurrence, the model has no inherent sense of word order.
    *   To solve this, we add a "Positional Encoding" vector to each input word's embedding. This vector gives the model information about the position of the word in the sequence. The original paper used sine and cosine functions of different frequencies.

### 🧠 Math & Stats Focus: Self-Attention

Self-attention is just the attention mechanism from Day 5, but applied in a special way.

*   **Query, Key, and Value come from the same place:**
    *   In the encoder's self-attention, the Query, Key, and Value vectors are all linear projections of the **same input sequence**.
    *   **In other words:** Each word creates a Query and "looks at" all the other words in the same sentence (which provide the Keys and Values).
    *   This allows the model to learn the internal relationships of the input sentence. For example, when processing the word "it" in "The cat drank the milk because it was thirsty", self-attention can learn to associate "it" with "cat".

*   **Multi-Head Attention:**
    *   Instead of just one set of Q, K, V projections, the Transformer does it multiple times in parallel (e.g., 8 "heads").
    *   Each "head" can learn a different type of relationship (e.g., one head might learn subject-verb relationships, another might learn pronoun references).
    *   The outputs of all the heads are concatenated and projected back to the expected dimension, creating a very rich representation.

### 📜 Key Research Paper

This is arguably the most important deep learning paper of the last decade.

*   **Paper:** "Attention Is All You Need" (Vaswani et al., 2017)
*   **Link:** [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   **Contribution:** Introduced the Transformer architecture. By demonstrating that a model based solely on attention could outperform RNN-based models with attention, it revolutionized the field. Its parallelizable nature unlocked the ability to train on vastly larger datasets, directly leading to the LLM era.

### 💻 Project: Build a Positional Encoding Function

The Transformer's self-attention layer is complex to build from scratch, but the positional encoding function is a great, self-contained coding exercise.

1.  **Read the formula:** In the "Attention Is All You Need" paper, find section 3.5. The formulas for PE are given there.
    *   `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
    *   `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
2.  **Implement in Python/NumPy:** Write a function that takes two arguments: `sequence_length` and `embedding_dim` (`d_model`).
3.  **The function should return a NumPy array of shape `(sequence_length, embedding_dim)`**.
4.  **Visualize the result:** Use `matplotlib` to plot the resulting positional encoding matrix. You should see a pattern of sine and cosine waves across the dimensions, which gives each position a unique signature.

### ✅ Progress Tracker

*   [ ] I can explain why recurrence is a bottleneck for processing long sequences.
*   [ ] I can describe the difference between "self-attention" and the "cross-attention" used in the original Seq2Seq models.
*   [ ] I understand the purpose of Positional Encodings.
*   [ ] I have read the abstract and introduction of "Attention Is All You Need".
