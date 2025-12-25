---
sidebar_position: 2
id: day-1
title: 'Day 1: The Genesis of Generative AI'
---

## Day 1: The Genesis of Generative AI & Foundational Concepts

### Objective

Understand the core concept of Generative AI, differentiate it from Discriminative AI, and grasp the fundamental probability concepts that form its bedrock.

### Core Concepts

*   **What is Generative AI?**
    *   Generative models learn the underlying distribution of data (`P(x)`) to generate new, synthetic data samples.
    *   In contrast, Discriminative models learn the boundary between classes (`P(y|x)`) to classify or predict.
    *   **Example:** A generative model could create a new image of a cat, while a discriminative model would classify an existing image as "cat" or "not a cat".

*   **A Brief History:**
    *   Early statistical models (Markov Chains) generating simple text.
    *   The introduction of Generative Adversarial Networks (GANs) in 2014 marked a significant leap in image generation.
    *   The rise of the Transformer architecture in 2017 paved the way for Large Language Models.

### 🧠 Math & Stats Focus: Probability Fundamentals

Generative AI is all about modeling probability distributions. Today's focus is on the basics.

*   **Random Variables:** Variables whose values are numerical outcomes of a random phenomenon (e.g., the result of a dice roll).
*   **Probability Distributions:** A function that describes the likelihood of obtaining the possible values that a random variable can assume.
    *   **Probability Mass Function (PMF):** For discrete variables (e.g., the probability of a specific word appearing).
    *   **Probability Density Function (PDF):** For continuous variables (e.g., the probability of a pixel having a certain intensity).
*   **Joint Probability `P(A, B)`:** The probability of two events occurring together.
*   **Conditional Probability `P(A|B)`:** The probability of event A occurring given that event B has already occurred. This is the foundation of models that generate text one word at a time.
    `P(A|B) = P(A, B) / P(B)`

### 📜 Key Research Paper

While many foundational papers exist, a good conceptual starting point is to understand the framework of probabilistic modeling.

*   **Paper:** "A New Framework for Machine Learning" (A Conceptual Introduction)
*   **Contribution:** This isn't a specific paper, but the core idea is that we can frame machine learning problems as finding the parameters `θ` of a model that best explain the data. For generative models, this is often done by maximizing the likelihood of the data, `P(X|θ)`.

### 💻 Project: A Simple Text Generator with Markov Chains

Build a character-level text generator using a Markov chain. This is a simple generative model that predicts the next character based only on the current character.

1.  **Get a text corpus:** Find a plain text file (e.g., a book from Project Gutenberg).
2.  **Build a transition table (dictionary):** For each character in the text, store a list of all characters that follow it. This is your learned probability distribution `P(next_char | current_char)`.
3.  **Generate text:**
    *   Pick a random starting character.
    *   Look up the current character in your transition table and randomly choose one of the possible next characters.
    *   Append the chosen character to your generated text and repeat the process.

### ✅ Progress Tracker

*   [ ] I can explain the difference between Generative and Discriminative AI.
*   [ ] I understand the concepts of PMF/PDF and conditional probability.
*   [ ] I have built a working character-level Markov chain text generator.
