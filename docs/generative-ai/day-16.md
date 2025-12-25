---
sidebar_position: 17
id: day-16
title: 'Day 16: Evaluating Generative Models - How Good is "Good"?'
---

## Day 16: Evaluating Generative Models - How Good is "Good"?

### Objective

Understand the unique challenges of evaluating generative models and learn about the key metrics used for both text and image generation.

### Core Concepts

*   **The Challenge of Evaluation:**
    *   Unlike discriminative models where you can easily measure accuracy, evaluating generative models is difficult. What does it mean for a generated sentence or image to be "correct"?
    *   We need metrics that can measure two key aspects:
        1.  **Fidelity/Quality:** How realistic and high-quality are the individual generated samples?
        2.  **Diversity:** How well does the generator capture the full range of variations present in the real data? (i.e., is it suffering from mode collapse?)

*   **Metrics for Text Generation:**
    *   **Perplexity (Revisited):** Still a useful metric. A lower perplexity on a held-out test set indicates the model is good at predicting realistic word sequences. However, it doesn't always correlate well with human judgment of creativity or coherence.
    *   **BLEU (Bilingual Evaluation Understudy):**
        *   Primarily used for machine translation, but adapted for other tasks like summarization.
        *   It measures the overlap of n-grams between the model's output and a set of human-written reference translations.
        *   **Limitation:** It focuses on precision (did the n-grams in the output appear in the reference?) but can penalize creativity and rephrasing.
    *   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**
        *   Similar to BLEU, but focuses on recall (did the n-grams from the reference appear in the output?).
        *   Better for summarization tasks where capturing key information is important.

*   **Metrics for Image Generation:**
    *   **Inception Score (IS):**
        *   Measures both quality and diversity. It uses a pre-trained Inception model to classify the generated images.
        *   **High Quality:** If the images are clear, the Inception model should be very confident in its classification (low entropy in the probability distribution for each image).
        *   **High Diversity:** If the generator produces a wide variety of images (e.g., all 1000 ImageNet classes), the overall distribution of class predictions should be uniform (high entropy).
        *   A higher Inception Score is better.
    *   **Fréchet Inception Distance (FID):
        *   The current standard metric. It improves upon IS.
        *   It compares the statistical distribution of features from a deep layer of a pre-trained Inception model for a batch of real images vs. a batch of generated images.
        *   Specifically, it models the distributions as multivariate Gaussians and calculates the "Fréchet distance" between them.
        *   **Intuition:** It measures how "far" the distribution of fake images is from the distribution of real images. **A lower FID score is better.**

### 🧠 Math & Stats Focus: Fréchet Distance

*   **Multivariate Gaussian Distribution:** A generalization of the bell curve to multiple dimensions. It's defined by a mean vector (`μ`) and a covariance matrix (`Σ`).
*   **Fréchet Distance:** A measure of the distance between two curves or distributions. For two multivariate Gaussian distributions (Real: `(μ_r, Σ_r)`, Fake: `(μ_f, Σ_f)`), the formula is:
    `d^2 = ||μ_r - μ_f||^2 + Tr(Σ_r + Σ_f - 2 * (Σ_r * Σ_f)^(1/2))`
    *   `||μ_r - μ_f||^2`: The squared distance between the mean vectors. This measures if the average features are different.
    *   `Tr(...)`: The trace of the matrix (sum of diagonal elements). This part measures if the spread and correlation of features are different.
    *   You don't need to memorize the formula, but you should understand that FID compares both the **mean** and the **covariance** of the image features, making it a robust metric.

### 📜 Key Research Paper

*   **Paper:** "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (Heusel et al., 2017)
*   **Link:** [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)
*   **Contribution:** While the title is about GAN training, this paper is famous for introducing the **Fréchet Inception Distance (FID)**. It quickly became the default metric for evaluating image generation models because it was shown to correlate much better with human judgment of image quality and diversity than the previous Inception Score.

### 💻 Project: Calculate FID for Your GAN

Actually calculating FID is complex as it requires running a specific Inception model version and having a large number of images. A more practical project is to use a pre-built library to compare two sets of images.

1.  **Generate two sets of images:**
    *   Use the simple GAN you built for MNIST (Day 13).
    *   Let it train for only a few epochs (e.g., 5) and save 1,000 generated images. These will be your "low quality" images.
    *   Let it train for many epochs (e.g., 50) and save another 1,000 images. These will be your "high quality" images.
2.  **Install the library:** `pip install pytorch-fid`. (Note: This library uses PyTorch, but it can be used from the command line on your saved image files regardless of how they were generated).
3.  **Get real images:** Save 1,000 real MNIST images to a separate folder.
4.  **Calculate FID scores:** Run the FID calculation from your command line.
    *   `python -m pytorch_fid /path/to/real_images /path/to/low_quality_images`
    *   `python -m pytorch_fid /path/to/real_images /path/to/high_quality_images`
5.  **Compare the scores:** Which FID score is lower? It should be the one for the "high quality" images, demonstrating that the FID score improves as your generator gets better.

### ✅ Progress Tracker

*   [ ] I can explain the difference between measuring "fidelity" and "diversity" in generative models.
*   [ ] I know that BLEU/ROUGE are used for text and IS/FID are used for images.
*   [ ] I can explain intuitively what FID measures (the distance between the distributions of real and fake image features).
*   [ ] I have used a library to calculate the FID score between two sets of images.
