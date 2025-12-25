---
sidebar_position: 16
id: day-15
title: 'Day 15: Multimodality & The CLIP Model'
---

## Day 15: Multimodality & The CLIP Model - Connecting Text and Images

### Objective

Understand the concept of multimodality and study the CLIP model, which learns a shared representation space for both text and images, enabling powerful new capabilities like zero-shot image classification.

### Core Concepts

*   **Modality:** A type or format of data. Text is one modality, images are another, audio is a third.
*   **Multimodal Models:** Models that are designed to process and relate information from two or more different modalities.
*   **The Goal:** To build a model that understands the *relationship* between, for example, the image of a dog and the text "a photo of a dog".

*   **CLIP (Contrastive Language-Image Pre-training):**
    *   CLIP is a model from OpenAI that learns a rich, shared space for text and images.
    *   **The Training Process:**
        1.  It's trained on a massive dataset of 400 million (image, text) pairs scraped from the internet.
        2.  It has two separate encoders: a **Text Encoder** (a Transformer) and an **Image Encoder** (e.g., a Vision Transformer - ViT).
        3.  During training, a batch of `N` (image, text) pairs is processed. The `N` images go through the Image Encoder, and the `N` text captions go through the Text Encoder, resulting in `N` image embeddings and `N` text embeddings.
        4.  **The Contrastive Objective:** The model's goal is to predict which of the `N x N` possible pairings in the batch are the correct ones. For a given image embedding, the cosine similarity to its *correct* text embedding should be maximized, while its similarity to all `N-1` *incorrect* text embeddings should be minimized.

*   **Zero-Shot Image Classification:**
    *   This is the magical capability that emerges from the contrastive training.
    *   To classify a new image, you don't need to fine-tune the model.
    *   **The Process:**
        1.  Take your new image and pass it through the Image Encoder to get its embedding.
        2.  Create text prompts for all your possible classes (e.g., "a photo of a dog", "a photo of a cat", "a photo of a car").
        3.  Pass these text prompts through the Text Encoder to get their embeddings.
        4.  Calculate the cosine similarity between the image embedding and each of the text embeddings.
        5.  The text prompt with the highest similarity is your predicted class!

### 🧠 Math & Stats Focus: Contrastive Learning and Cosine Similarity

*   **Contrastive Loss:** The loss function that powers CLIP. It aims to pull "positive pairs" (the correct image and text) together in the embedding space while pushing "negative pairs" (all other combinations) apart.
    *   Given an image `I_1` and its text `T_1`, and another image `I_2` and its text `T_2`.
    *   The model wants to make `Similarity(Embedding(I_1), Embedding(T_1))` high.
    *   The model wants to make `Similarity(Embedding(I_1), Embedding(T_2))` low.
    *   This is done across a large batch, creating a matrix of similarities, where the model is trained to maximize the values on the diagonal (the correct pairs).

*   **Cosine Similarity:** Revisited from Day 3. It is the perfect metric for this task.
    `Similarity(v, w) = (v · w) / (||v|| * ||w||)`
    *   Because CLIP normalizes its embeddings to have a length of 1, the cosine similarity is simply the dot product of the two embedding vectors. This makes the computation very efficient.

### 📜 Key Research Paper

*   **Paper:** "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021 - The "CLIP" paper)
*   **Link:** [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
*   **Contribution:** CLIP demonstrated that it was possible to achieve state-of-the-art zero-shot classification performance by training on a massive, noisy dataset of (image, text) pairs from the web, without needing a traditional, human-labeled dataset like ImageNet. It provides a powerful, flexible bridge between vision and language and is a key component in many modern generative models like DALL-E 2 and Stable Diffusion (which uses it to condition the diffusion process on a text prompt).

### 💻 Project: Perform Zero-Shot Classification with CLIP

Use the pre-trained CLIP model from Hugging Face to build a flexible, zero-shot image classifier.

1.  **Install Libraries:** `pip install transformers Pillow requests`.
2.  **Load a Pre-trained CLIP Model and Processor:**
    *   `from transformers import AutoProcessor, AutoModel`
    *   `processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")`
    *   `model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")`
3.  **Get an Image:** Find any image URL online.
4.  **Define Your Classes:** Create a list of text labels for whatever you want to classify. This is completely flexible!
    *   `labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]`
5.  **Process the Inputs:** Use the `processor` to prepare both the image and the text labels. It will handle tokenization, image resizing, and normalization for you.
6.  **Get Embeddings:**
    *   Pass the processed image through the model to get the image embedding.
    *   Pass the processed text through the model to get the text embeddings.
7.  **Calculate Similarities and Predict:**
    *   Use the text and image embeddings to calculate the logits (dot product similarities).
    *   Apply a `softmax` to the logits to get the probabilities.
    *   The label with the highest probability is your prediction.
8.  **Experiment:** Try a completely different image and a new, unrelated set of labels (e.g., "a photo of a mountain", "a photo of a beach"). The same model should work without any retraining.

*The Hugging Face documentation has an excellent guide and code snippet for [Zero-shot image classification with CLIP](https://huggingface.co/docs/transformers/tasks/zero_shot_image_classification) that is perfect for this project.*

### ✅ Progress Tracker

*   [ ] I can define what a "multimodal" model is.
*   [ ] I can explain the high-level goal of CLIP's contrastive training.
*   [ ] I understand the process of "zero-shot image classification".
*   [ ] I have used a pre-trained CLIP model to classify an image using custom text labels.
