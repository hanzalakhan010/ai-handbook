---
sidebar_position: 27
id: day-26
title: 'Day 26: Hallucinations and Grounding - Keeping LLMs Factual'
---

## Day 26: Hallucinations and Grounding - Keeping LLMs Factual

### Objective

Understand why Large Language Models "hallucinate," the different types of hallucinations, and advanced techniques (beyond basic RAG) for grounding their responses in factual, verifiable information.

### Core Concepts

*   **What is an LLM Hallucination?**
    *   An LLM hallucination is a generated response that is factually incorrect, nonsensical, or unfaithful to the provided source content, even though it may sound plausible and confident.
    *   It's not "lying"; it's the model confidently generating text that doesn't correspond to reality or its input.

*   **Why Do LLMs Hallucinate?**
    1.  **Mismatch between Training & Inference:** LLMs are trained to predict the next most probable token based on their training data. This data contains biases, inconsistencies, and may not reflect current reality. During inference, they continue this pattern.
    2.  **Lack of Real-World Knowledge:** They don't "know" facts in the human sense. They only process patterns in text.
    3.  **Ambiguous Prompts:** Vague prompts give the model too much freedom to generate.
    4.  **Data Cut-off:** Their knowledge is static at training time.
    5.  **Output Length/Complexity:** Longer and more complex generations increase the chance of errors.
    6.  **Confidence vs. Accuracy:** A high probability for a token doesn't mean it's factually correct, just that it frequently co-occurs in the training data.

*   **Types of Hallucinations:**
    *   **Factually Incorrect:** Generating false statements about people, events, or objects.
    *   **Contradictory:** Generating information that contradicts the provided context.
    *   **Invented Information:** Making up sources, dates, or details.
    *   **Nonsensical:** Producing grammatically correct but logically absurd statements.

*   **Grounding Techniques (Beyond Basic RAG):**
    *   **Advanced RAG (Day 18):**
        *   **Query Expansion:** Rephrase the user's query multiple times to get better retrieval.
        *   **Re-ranking:** After initial retrieval, use a smaller, more powerful model to re-rank the retrieved documents for relevance.
        *   **Fine-tuned Retrieval:** Fine-tune the embedding model for your specific knowledge domain.
    *   **Self-Correction / Self-Refinement:**
        *   The LLM generates a response, then *critiques its own response* based on the retrieved documents or a set of rules.
        *   It then re-generates a better response. This often involves multiple passes through the LLM.
    *   **Fact-Checking Tools (Tool Use):**
        *   Integrate external fact-checking APIs or knowledge graphs (like Wikipedia APIs, Google Search APIs) as tools for the agent (Day 20) to use to verify its own statements.
    *   **Knowledge Graphs:**
        *   Representing factual information as a graph of entities and relationships. LLMs can be prompted to query these graphs to retrieve accurate information directly.

### 🧠 Math & Stats Focus: Uncertainty Quantification (Conceptual)

While difficult, quantifying uncertainty in LLM outputs is an active area of research to combat hallucinations.

*   **Entropy in Output Distribution:** A model's confidence in its next token prediction can be measured by the entropy of its output probability distribution (softmax). A low entropy (peaky distribution) suggests high confidence, but doesn't guarantee factual accuracy.
*   **Calibration:** The ideal scenario is that if a model predicts something with 80% probability, it should be correct 80% of the time. LLMs are often poorly calibrated, meaning their stated confidence doesn't match their actual accuracy. Research aims to improve this.
*   **Multiple Samples & Consensus:** Generating multiple responses (e.g., using different random seeds or sampling parameters) and looking for consensus can give an indication of how "sure" the model is about a fact. If all generated responses agree, it's more likely to be correct.

### 📜 Key Research Paper

*   **Paper:** "Survey of Hallucination in Large Language Models" (Ji et al., 2023)
*   **Link:** [https://arxiv.org/abs/2303.01042](https://arxiv.org/abs/2303.01042)
*   **Contribution:** This comprehensive survey provides a detailed overview of the different types of LLM hallucinations, their causes, and the various techniques proposed to detect and mitigate them. It's an excellent resource for anyone wanting to delve deeper into this critical problem.

### 💻 Project: Implement Simple Self-Correction

Your goal is to implement a basic form of self-correction for an LLM's response.

1.  **Choose an LLM:** Use a publicly available LLM (via API or Hugging Face `pipeline`) that supports chat-like interactions.
2.  **Generate a Potentially Hallucinatory Response:**
    *   Give the LLM a prompt that it might struggle with or where it's prone to hallucination.
    *   **Example Prompt:** "Who was the founder of the first company that put a human on Mars?" (This is a trick question; no human has been to Mars yet).
    *   Save the initial response.
3.  **Implement a Critique Prompt:**
    *   Now, create a second prompt that takes the original prompt and the LLM's initial response, and asks the LLM to critique its own answer for factual accuracy.
    *   **Critique Prompt:**
        ```
        Original Question: "Who was the founder of the first company that put a human on Mars?"
        Model's Answer: "[LLM's initial answer here]"

        Review this answer for factual accuracy. Identify any incorrect statements or hallucinations. If the answer is incorrect, provide a corrected answer based on verifiable facts.
        ```
4.  **Get the Critique:** Send this critique prompt to the LLM.
5.  **Analyze the Improvement:** Does the LLM identify its own hallucination? Does it correctly state that no human has been to Mars? This simple technique can significantly improve the factual correctness of outputs.

### ✅ Progress Tracker

*   [ ] I can define LLM hallucination and list at least 3 reasons why it occurs.
*   [ ] I can describe how advanced RAG techniques and self-correction help mitigate hallucinations.
*   [ ] I understand the conceptual role of uncertainty quantification in detecting hallucinations.
*   [ ] I have implemented a basic self-correction mechanism for an LLM.
