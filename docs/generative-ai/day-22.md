---
sidebar_position: 23
id: day-22
title: 'Day 22: Multimodal LLMs & Instruction Following'
---

## Day 22: Multimodal LLMs & Instruction Following - Beyond Text

### Objective

Explore how LLMs are extended to become multimodal, capable of understanding and generating content across different data types (text, images, audio), and the impact of this on instruction following.

### Core Concepts

*   **The Rise of Multimodality:**
    *   Traditional LLMs (like GPT-3) are text-in, text-out.
    *   Models like CLIP (Day 15) connected text and images by learning a shared embedding space.
    *   **Multimodal LLMs** take this a step further: they can *take multimodal input* (e.g., an image *and* a text prompt) and *generate multimodal output* (e.g., generate text describing an image, or generate an image from a text prompt).

*   **How Multimodal LLMs Work (High-Level Architectures):**
    *   **Shared Encoder:** Some models use separate encoders for each modality (e.g., a ViT for images, a Transformer for text) and then project their embeddings into a shared space (like CLIP). An LLM then sits on top of this shared representation.
    *   **Unified Transformer:** More advanced architectures try to unify the processing within a single Transformer. They might convert images/audio into "tokens" that look like text tokens (e.g., image patches from ViT, or spectrograms for audio) and feed them into a large Transformer alongside text tokens. The model then learns to predict the next token, regardless of modality.

*   **Key Capabilities of Multimodal LLMs:**
    *   **Image Captioning:** "Describe this image."
    *   **Visual Question Answering (VQA):** "What color is the car in this picture?"
    *   **Text-to-Image Generation (Advanced):** Generate images from rich textual descriptions (e.g., DALL-E 2, Stable Diffusion, Midjourney). The LLM often acts as the "understanding" component that converts the prompt into a suitable representation for the image generation model.
    *   **Visual Chatbots:** Engage in conversation about an image.

*   **Connecting to Instruction Following:**
    *   Just like text-only LLMs needed instruction tuning and RLHF to become helpful assistants, multimodal LLMs undergo similar processes.
    *   They are trained on datasets of (multimodal instruction, multimodal response) pairs, teaching them to follow instructions that involve different modalities.
    *   **Example Instruction:** "Describe the main object in this image and tell me its color." (Input: Image + Text Instruction; Output: Text Description).

### 🧠 Math & Stats Focus: Cross-Attention in Multimodality

*   **Cross-Attention (Revisited):** This concept (from the original Transformer's decoder) is fundamental to multimodal LLMs.
    *   It allows one modality to "query" information from another modality.
    *   **Example:** When generating a text description for an image:
        *   The **Query (Q)** comes from the text decoder's hidden state (representing what text it's trying to generate).
        *   The **Keys (K)** and **Values (V)** come from the image encoder's output (representing the different visual features in the image).
        *   The text decoder can then attend to the most relevant visual features in the image to generate the appropriate words.
    *   This is how models can intelligently integrate information across modalities.

### 📜 Key Research Paper

There are many multimodal LLM papers. Here are two influential ones:

*   **Paper:** "Flamingo: a Visual Language Model for Few-Shot Learning" (Alayrac et al., 2022)
*   **Link:** [https://arxiv.org/abs/2204.14198](https://arxiv.org/abs/2204.14198)
*   **Contribution:** Flamingo was one of the first highly successful large visual language models. It used a pre-trained LLM and connected it to a pre-trained vision encoder via cross-attention layers, allowing it to achieve strong few-shot performance on a wide range of multimodal tasks. It demonstrated how to effectively combine existing powerful models to create multimodal intelligence.

*   **Paper:** "GPT-4 Technical Report" (OpenAI, 2023)
*   **Link:** [https://arxiv.org/abs/2303.08774](https://arxiv.org/abs/2303.08774)
*   **Contribution:** While not detailing the *architecture*, this report showed the unprecedented capabilities of a truly large-scale multimodal model (GPT-4 with vision capabilities). It demonstrated the ability to take image inputs and follow complex instructions involving both visual and linguistic understanding, pushing the boundaries of what's possible with multimodal AI.

### 💻 Project: Use a Multimodal Model for Visual Question Answering (VQA)

Use a pre-trained multimodal model from Hugging Face to answer questions about an image.

1.  **Install Libraries:** `pip install transformers Pillow requests`.
2.  **Load a Pre-trained VQA Model and Processor:**
    *   `from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering`
    *   `processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")`
    *   `model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")`
3.  **Get an Image:** Find an image URL online that contains something you can ask a question about (e.g., a picture of a cat, a city scene).
4.  **Ask a Question:**
    *   `question = "What is in the image?"`
    *   `question = "How many people are there?"`
5.  **Process and Predict:**
    *   Use the `processor` to prepare the image and question.
    *   Pass the processed inputs to the `model` to get logits.
    *   The model will output a probability distribution over possible answers. Decode the most likely answer.
6.  **Experiment:** Try different images and different types of questions. How well does it perform? Does it struggle with counting? Does it understand abstract concepts?

### ✅ Progress Tracker

*   [ ] I can define what a multimodal LLM is.
*   [ ] I understand the role of cross-attention in connecting different modalities.
*   [ ] I can list at least two capabilities of multimodal LLMs.
*   [ ] I have used a pre-trained multimodal model to answer a question about an image.
