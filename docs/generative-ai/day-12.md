---
sidebar_position: 13
id: day-12
title: 'Day 12: Reinforcement Learning with Human Feedback (RLHF)'
---

## Day 12: Reinforcement Learning with Human Feedback (RLHF)

### Objective

Understand the three-step process of RLHF, the technique used to align language models with human preferences and make them safer, more helpful, and less prone to generating harmful content.

### Core Concepts

*   **The Problem with Instruction Tuning:**
    *   Instruction tuning (Day 11) makes a model good at following instructions, but it doesn't guarantee the *quality* of the response.
    *   How do we teach the model to be helpful, truthful, and harmless? It's hard to create a dataset that explicitly defines these qualities. It's much easier for a human to say "this response is better than that one."

*   **RLHF: The Big Idea:**
    *   Use Reinforcement Learning (RL) to optimize the language model directly based on human preferences.
    *   The "environment" is the user's prompt, the "agent" is the LLM, the "action" is the generated response, and the "reward" is a score indicating how "good" the response is according to a human.

*   **The Three-Step Process:**
    1.  **Step 1: Supervised Fine-Tuning (SFT):**
        *   This is exactly the same as instruction tuning. You start with a base LLM and fine-tune it on a dataset of high-quality (instruction, response) pairs written by human labelers. This gives you a good starting point.
    2.  **Step 2: Training a Reward Model (RM):**
        *   This is the key to capturing human preferences.
        *   Take a prompt and generate several different responses from the SFT model.
        *   A human labeler then **ranks** these responses from best to worst.
        *   You then train a separate model (the Reward Model) that takes a (prompt, response) pair as input and outputs a single scalar "reward" score. The RM is trained to give higher scores to the responses that humans ranked higher.
    3.  **Step 3: Reinforcement Learning Optimization:**
        *   The SFT model is now optimized using an RL algorithm (commonly **PPO - Proximal Policy Optimization**).
        *   **The Loop:**
            a. A prompt is sampled from the dataset.
            b. The SFT model (the "policy") generates a response.
            c. The Reward Model from Step 2 evaluates the response and gives it a reward score.
            d. This reward is used to update the weights of the SFT model via the PPO algorithm.
        *   A **KL-divergence** penalty is also added to prevent the model from straying too far from the original SFT model, ensuring it doesn't "over-optimize" for the reward and start generating nonsensical but high-reward text.

### 🧠 Math & Stats Focus: The Reward Model and KL-Divergence

*   **The Reward Model's Loss Function:** The RM is trained on pairs of ranked responses. For a given prompt, if response A was ranked higher than response B, the RM's loss function is designed to maximize the difference between their predicted scores: `loss = -log(sigmoid(score(A) - score(B)))`. This is a "pairwise ranking loss."

*   **KL-Divergence:** A measure of how one probability distribution `P` diverges from a second expected probability distribution `Q`.
    `D_KL(P || Q) = Σ P(x) log(P(x) / Q(x))`
    *   In RLHF, it's used as a penalty term in the reward function. `P` is the output distribution of the RL-tuned model, and `Q` is the output distribution of the original SFT model.
    *   **Reward_final = Reward_RM - β * D_KL(P_RL || Q_SFT)**
    *   This penalty ensures that the RL-tuned model doesn't generate text that the original SFT model would have considered extremely unlikely. It keeps the model "on-distribution" and prevents it from saying gibberish just to get a high reward.

### 📜 Key Research Paper

*   **Paper:** "Training language models to follow instructions with human feedback" (Ouyang et al., 2022 - The "InstructGPT" paper)
*   **Link:** [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
*   **Contribution:** This paper from OpenAI detailed the full RLHF process and applied it to create "InstructGPT," the predecessor to ChatGPT. It showed that this alignment process was incredibly effective at making models more helpful and truthful, even allowing smaller 1.3B parameter models to be preferred by users over the much larger 175B parameter GPT-3 base model. This was the blueprint for modern chatbot alignment.

### 💻 Project: Rank Model Responses

You don't need a huge team of labelers to understand Step 2 of RLHF. You can do it yourself.

1.  **Pick a Prompt:** Choose an open-ended instruction. For example: "Explain the theory of relativity to a 5-year-old."
2.  **Generate Multiple Responses:** Use a generative model (like the GPT-2 from Day 8) to generate 3-4 different responses to this prompt. To get different responses, you can use sampling (`do_sample=True`) and change the `top_k` or `top_p` parameters.
3.  **Act as the Human Labeler:** Read the responses you generated.
4.  **Rank Them:** Order the responses from best to worst based on your own preferences.
5.  **Justify Your Ranking:** For each response, write a short sentence explaining *why* you ranked it that way. What makes the best response good? (e.g., "Uses a simple analogy"). What makes the worst response bad? (e.g., "Too technical," "Factually incorrect," "Didn't answer the question").

This exercise puts you in the shoes of a human labeler and forces you to think critically about the subtle qualities that make one generated response better than another. This is the core data used to train the Reward Model.

### ✅ Progress Tracker

*   [ ] I can name the three main steps of the RLHF process.
*   [ ] I can explain the purpose of the Reward Model.
*   [ ] I understand why a KL-divergence penalty is needed during the RL optimization step.
*   [ ] I have ranked several model responses for a given prompt and justified my ranking.
