---
sidebar_position: 5
id: day-4
title: 'Day 4: Sequence-to-Sequence Models'
---

## Day 4: Sequence-to-Sequence Models & The Encoder-Decoder Architecture

### Objective

Understand the sequence-to-sequence (Seq2Seq) framework and the encoder-decoder architecture, which form the basis for many generative tasks like machine translation and summarization.

### Core Concepts

*   **Sequence-to-Sequence (Seq2Seq):**
    *   A type of model that takes a sequence of items (words, images, etc.) as input and generates another sequence of items as output.
    *   The input and output sequences can be of different lengths.
    *   **Examples:**
        *   Machine Translation: (Input: "Hello, how are you?", Output: "Bonjour, comment ça va?")
        *   Summarization: (Input: A long article, Output: A short summary)
        *   Chatbots: (Input: A user's question, Output: A generated answer)

*   **The Encoder-Decoder Architecture:**
    *   The standard architecture for Seq2Seq models. It consists of two main components:
    *   **1. The Encoder:**
        *   A neural network (traditionally a Recurrent Neural Network - RNN or LSTM) that reads the entire input sequence.
        *   Its job is to compress the information from the input sequence into a single fixed-size vector, often called the **context vector** or "thought vector".
        *   This vector is the final hidden state of the encoder.
    *   **2. The Decoder:**
        *   Another neural network (also typically an RNN or LSTM) that takes the encoder's context vector as its initial hidden state.
        *   Its job is to generate the output sequence, one item at a time. At each step, it takes the previously generated item as input to help decide the next one.

*   **The Bottleneck Problem:**
    *   The fixed-size context vector is a major limitation. The encoder must cram all the information from a potentially very long input sentence into this single vector.
    *   This makes it difficult for the model to handle long sequences, as information from the beginning of the sequence can be lost by the time the encoder finishes. This is the problem that "Attention" will solve.

### 🧠 Math & Stats Focus: Recurrent Neural Networks (RNNs)

The original encoder-decoder models were built with RNNs.

*   **Recurrence Relation:** An RNN processes a sequence by iterating through the items and applying a recurrence relation at each step `t`:
    `h_t = f(W * x_t + U * h_{t-1})`
    *   `h_t`: The hidden state (memory) at the current time step `t`.
    *   `x_t`: The input at the current time step `t` (e.g., a word vector).
    *   `h_{t-1}`: The hidden state from the previous time step.
    *   `W` and `U`: Weight matrices that are learned during training. They are the **same** for all time steps.
    *   `f`: A non-linear activation function (like `tanh`).
*   **Vanishing/Exploding Gradients:** A major problem with simple RNNs. When backpropagating errors through many time steps, the gradients can become extremely small (vanish) or extremely large (explode), making it very difficult to learn long-range dependencies. This is why more complex variants like LSTMs and GRUs were developed.

### 📜 Key Research Paper

*   **Paper:** "Sequence to Sequence Learning with Neural Networks" (Sutskever, Vinyals, & Le, 2014)
*   **Link:** [https://arxiv.org/abs/1409.3215](https://arxiv.org/abs/1409.3215)
*   **Contribution:** This paper, along with a similar one by Cho et al., introduced the general Encoder-Decoder architecture using LSTMs. They demonstrated its effectiveness on machine translation, showing that a purely neural approach could outperform traditional phrase-based systems. It laid the groundwork for the attention mechanism.

### 💻 Project: Build a Simple Encoder-Decoder Model

Build a character-level encoder-decoder model to "translate" a sequence. A fun example is to learn to reverse a sequence of characters.

1.  **Generate Data:** Create pairs of strings, e.g., (Input: "hello", Output: "olleh").
2.  **Vectorize Data:** Convert your characters into one-hot encoded vectors.
3.  **Build the Encoder:**
    *   Use a Keras `Input` layer.
    *   Use an `LSTM` layer. Discard the per-step outputs and only keep the final hidden state (`return_state=True`). These final states are your "context vector".
4.  **Build the Decoder:**
    *   The decoder `LSTM` will take the context vector from the encoder as its `initial_state`.
    *   At each step, the decoder's output will be passed through a `Dense` layer with a `softmax` activation to predict the next character.
5.  **Train the Model:** Train the model to predict the reversed sequence.
6.  **Test:** Give the model a new sequence and see if it can generate the reversed version correctly.

*This is a non-trivial project. Following a Keras tutorial like [this one](https://keras.io/examples/nlp/lstm_seq2seq/) is highly recommended to understand the implementation details.*

### ✅ Progress Tracker

*   [ ] I can describe the roles of the encoder and the decoder in a Seq2Seq model.
*   [ ] I understand the "bottleneck" problem of the fixed-size context vector.
*   [ ] I have a conceptual understanding of how an RNN processes a sequence.
*   [ ] I have attempted to build or have read through the code for a simple Seq2Seq model.
