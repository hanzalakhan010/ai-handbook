---
sidebar_position: 3
id: day-2
title: 'Day 2: Language Models & The Power of Context'
---

## Day 2: Language Models & The Power of Context

### Objective

Understand what a language model is, how its performance is measured, and explore the limitations of early n-gram models.

### Core Concepts

*   **What is a Language Model?**
    *   A language model is a probability distribution over a sequence of words.
    *   Given a sequence of words, it can assign a probability to the entire sequence.
    *   More practically, it can predict the next word in a sequence: `P(w_n | w_1, w_2, ..., w_{n-1})`.

*   **N-Gram Models:**
    *   An early and simple type of language model that makes a simplifying assumption (the **Markov assumption**): the probability of the next word depends only on the previous `n-1` words.
    *   **Bigram Model (n=2):** `P(w_n | w_{n-1})`
    *   **Trigram Model (n=3):** `P(w_n | w_{n-2}, w_{n-1})`
    *   **How it works:** You count word co-occurrences in a large text corpus to estimate these conditional probabilities.
    *   **Limitation:** Fails to capture long-range dependencies and context. It also suffers from data sparsity (many n-grams will never appear in the training text).

### 🧠 Math & Stats Focus: Perplexity and Information Theory

How do we know if a language model is good?

*   **Chain Rule of Probability:** The probability of a sequence is the product of the conditional probabilities of each word.
    `P(w_1, ..., w_n) = P(w_1) * P(w_2|w_1) * ... * P(w_n|w_1, ..., w_{n-1})`

*   **Cross-Entropy:** A measure from information theory that quantifies the difference between two probability distributions. In language modeling, it measures how well the model's predicted distribution (`q`) matches the true distribution of words in the text (`p`). A lower cross-entropy is better.
    `H(p, q) = - Σ p(x) log(q(x))`

*   **Perplexity:** The standard metric for evaluating language models. It is the exponentiation of the cross-entropy.
    `Perplexity = 2^H(p,q)`
    *   **Intuition:** Perplexity is a measure of how "surprised" the model is by the next word. A perplexity of 100 means that on average, the model is as confused as if it had to choose between 100 different words at each step. **A lower perplexity is better.**

### 📜 Key Research Paper

The foundation of statistical language modeling was laid out in the late 80s and early 90s.

*   **Paper:** "A Maximum Entropy Approach to Natural Language Processing" (Berger et al., 1996)
*   **Link:** [https://aclanthology.org/J96-2004.pdf](https://aclanthology.org/J96-2004.pdf)
*   **Contribution:** This paper helped popularize the use of maximum entropy models (a form of logistic regression) for statistical NLP tasks, providing a more robust framework than simple n-gram counting and allowing for the inclusion of diverse features.

### 💻 Project: Build a Bigram Language Model

Implement a simple bigram (n=2) language model from scratch.

1.  **Get a text corpus:** Use a simple, clean text file.
2.  **Tokenize the text:** Split the text into a list of words (tokens). Add special `<s>` (start) and `</s>` (end) tokens to each sentence.
3.  **Count frequencies:**
    *   Count the occurrences of each individual word (unigrams).
    *   Count the occurrences of each pair of adjacent words (bigrams).
4.  **Calculate probabilities:** For any given word `w1`, the probability of the next word `w2` is `P(w2|w1) = count(w1, w2) / count(w1)`.
5.  **Generate text:**
    *   Start with the `<s>` token.
    *   Given the current word, look up its possible next words and their probabilities.
    *   Use these probabilities to sample the next word.
    *   Repeat until you sample an `</s>` token or reach a maximum length.
    *   How does the generated text compare to the character-level Markov chain?

### ✅ Progress Tracker

*   [ ] I can define a language model and its primary purpose.
*   [ ] I understand the Markov assumption behind n-gram models.
*   [ ] I can explain what Perplexity means intuitively.
*   [ ] I have built a working bigram language model.
