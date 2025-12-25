---
sidebar_position: 4
id: day-3
title: 'Day 3: Moving Beyond N-Grams - Neural Language Models'
---

## Day 3: Moving Beyond N-Grams - Neural Language Models

### Objective

Understand how neural networks can be used to create language models and learn about the concept of word embeddings, which revolutionized how we represent words.

### Core Concepts

*   **The Problem with N-Grams:**
    *   **Sparsity:** Many valid word sequences will not appear in the training data, leading to zero probabilities.
    *   **Storage:** The number of possible n-grams explodes as `n` and the vocabulary size increase.
    *   **No notion of similarity:** The model doesn't know that "cat" and "kitten" are similar. The n-grams `(the, cat, sat)` and `(the, kitten, sat)` are treated as completely different contexts.

*   **Neural Language Models:**
    *   Instead of counting, we train a neural network to predict the next word.
    *   **Input:** A sequence of words.
    *   **Output:** A probability distribution over the entire vocabulary for the next word.
    *   This approach solves the sparsity problem because the model can generalize from the training data.

*   **Word Embeddings (Word Vectors):**
    *   The key innovation. Instead of representing words as discrete IDs (e.g., "cat" is 534), we represent them as dense vectors in a continuous multi-dimensional space.
    *   **How it works:** The neural network learns a "lookup table" or "embedding matrix" where each row is the vector for a word. These vectors are learned as part of the training process.
    *   **The Magic:** Words with similar meanings end up with similar vectors. The model can now understand that "cat" and "kitten" are related, because their vectors will be close to each other in the embedding space. This solves the similarity problem of n-grams.

### 🧠 Math & Stats Focus: Vectors and Dot Products

Word embeddings are vectors. Understanding vector operations is key.

*   **Vector Space:** A collection of vectors. In our case, this is the "embedding space" where all our word vectors live.
*   **Vector:** A point in space, represented by a list of numbers (e.g., `[0.1, -0.4, 0.8, ...]`). The `dimension` of the embedding is the length of this list.
*   **Dot Product:** A measure of similarity between two vectors. If `v` and `w` are two word vectors, their dot product `v · w` is higher when they point in a similar direction. This is used extensively in attention mechanisms later on.
    `v · w = Σ v_i * w_i`
*   **Cosine Similarity:** A normalized version of the dot product that measures the cosine of the angle between two vectors. A value of 1 means they are identical, 0 means they are orthogonal (unrelated), and -1 means they are opposite.
    `Similarity(v, w) = (v · w) / (||v|| * ||w||)`

### 📜 Key Research Paper

This is one of the most influential papers in NLP history.

*   **Paper:** "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al., 2013)
*   **Link:** [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
*   **Contribution:** Introduced the **Word2Vec** model, a highly efficient way to learn high-quality word embeddings from massive amounts of text. It demonstrated that these embeddings capture semantic relationships, famously showing that `vector('King') - vector('Man') + vector('Woman')` results in a vector very close to `vector('Queen')`.

### 💻 Project: Explore Pre-trained Word Embeddings

You don't always have to train your own embeddings. Many pre-trained versions are available.

1.  **Load a pre-trained model:** Use a library like `gensim` to load a pre-trained Word2Vec or GloVe model. (GloVe is another popular embedding technique).
    *   `import gensim.downloader as api`
    *   `word_vectors = api.load("glove-wiki-gigaword-100")` (This will download the model).
2.  **Find similar words:** Use the `most_similar()` method to find words with similar vectors.
    *   `word_vectors.most_similar('king')`
    *   `word_vectors.most_similar('car')`
3.  **Perform analogies:** Use the `most_similar()` method with the `positive` and `negative` arguments to perform the famous "king - man + woman" analogy.
    *   `word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])`
4.  **Measure similarity:** Pick some word pairs (e.g., cat/kitten, cat/dog, cat/car) and calculate their cosine similarity using `word_vectors.similarity()`. Do the results match your intuition?

### ✅ Progress Tracker

*   [ ] I can explain why neural language models are an improvement over n-gram models.
*   [ ] I understand what a word embedding is and why it's powerful.
*   [ ] I can perform basic vector similarity and analogy tasks with a pre-trained model.
