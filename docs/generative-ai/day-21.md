---
sidebar_position: 22
id: day-21
title: 'Day 21: Vision Transformers (ViT) - Transformers for Images'
---

## Day 21: Vision Transformers (ViT) - Transformers for Images

### Objective

Understand how the Transformer architecture, originally designed for sequential data like text, can be adapted to process image data, leading to the **Vision Transformer (ViT)**.

### Core Concepts

*   **The Problem with CNNs (for Scale):**
    *   While CNNs (from Day 21 in the ML track) are excellent for images, they have some limitations. Their inductive biases (locality, translation equivariance) are great for images but might restrict their ability to learn very global relationships.
    *   As datasets grow massive (e.g., JFT-300M, a Google dataset with 300 million images), CNNs sometimes struggle to scale as effectively as Transformers do for text.

*   **ViT's Big Idea:**
    *   "Why not just treat images like text?"
    *   Break an image into small, fixed-size patches (like words in a sentence).
    *   Linearize these patches.
    *   Feed the sequence of patches into a standard Transformer Encoder.

*   **ViT Architecture (High-Level):**
    1.  **Image Patching & Linear Embedding:**
        *   An input image (e.g., 224x224 pixels) is split into a grid of non-overlapping patches (e.g., 16x16 pixels).
        *   Each patch is then flattened into a 1D vector.
        *   These flattened patch vectors are projected into a higher-dimensional embedding space using a linear layer.
    2.  **Class Token:**
        *   Similar to BERT's `[CLS]` token, a learnable "class token" embedding is prepended to the sequence of patch embeddings. The final state corresponding to this token is used for classification.
    3.  **Positional Embeddings:**
        *   Since the patches are now a sequence, we need to add positional information. Learnable 1D positional embeddings are added to the patch embeddings, just like in a text Transformer.
    4.  **Transformer Encoder:**
        *   The sequence of (class token + patch embeddings + positional embeddings) is fed into a standard Transformer Encoder stack (Multi-Head Self-Attention + Feed-Forward Networks).
    5.  **MLP Head:**
        *   The output corresponding to the class token from the Transformer Encoder is passed through a Multi-Layer Perceptron (MLP) head for classification.

### 🧠 Math & Stats Focus: Patch Embeddings

*   **Image to Sequence:** The core mathematical transformation in ViT is converting a 2D image into a 1D sequence of vectors.
    *   **Input Image:** `H x W x C` (Height, Width, Channels)
    *   **Patch Size:** `P x P`
    *   **Number of Patches:** `(H*W) / (P*P)`
    *   **Flattening:** Each `P x P x C` patch is flattened into a vector of size `P*P*C`.
    *   **Linear Projection:** This `P*P*C` vector is then mapped to a desired `D` dimension (the embedding size of the Transformer) using a weight matrix `W_p` and bias `b_p`:
        `Embedding = Patch_vector · W_p + b_p`
    *   The resulting sequence is `Number_of_Patches x D`.

### 📜 Key Research Paper

*   **Paper:** "An Image is Worth 16x16 Words: Transformers for Image Recognition At Scale" (Dosovitskiy et al., 2020)
*   **Link:** [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
*   **Contribution:** This paper from Google Brain introduced the Vision Transformer (ViT), directly challenging the dominance of CNNs in computer vision. It demonstrated that a purely Transformer-based model, when trained on sufficiently large datasets, could achieve state-of-the-art performance on image classification, outperforming even the best CNNs. This opened up a new era of applying Transformer architectures beyond NLP.

### 💻 Project: Use a Pre-trained ViT for Image Classification

You can use a pre-trained ViT model from Hugging Face for image classification, just like you did with BERT for text.

1.  **Install Libraries:** `pip install transformers torch torchvision`.
2.  **Load a Pre-trained ViT Model and Processor:**
    *   `from transformers import AutoImageProcessor, AutoModelForImageClassification`
    *   `processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")`
    *   `model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")`
3.  **Get an Image:** Find any image URL online (e.g., of a cat, dog, car).
4.  **Process the Image:**
    *   Use the `processor` to prepare the image. It will handle resizing, normalization, and converting it into the tensor format the model expects.
5.  **Make a Prediction:**
    *   Pass the processed image through the `model`.
    *   Get the logits (raw scores) and apply `softmax` to get probabilities.
    *   Use `model.config.id2label` to map the predicted class ID to a human-readable label.
6.  **Experiment:** Try different images. Does the model correctly identify common objects?

### ✅ Progress Tracker

*   [ ] I can explain the core idea of how ViT adapts Transformers for images.
*   [ ] I understand the steps involved in converting an image into a sequence of patches for a ViT.
*   [ ] I have a conceptual understanding of how ViT uses positional embeddings.
*   [ ] I have used a pre-trained ViT model to classify an image.
