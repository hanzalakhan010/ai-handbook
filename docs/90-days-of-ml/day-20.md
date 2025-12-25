---
id: day-20
title: 'Day 20: Model Deployment with Joblib and Flask'
---

## Day 20: Building Your First Neural Network with Keras (MNIST)

Today marks an exciting step as we delve into **Deep Learning** by building our very first Artificial Neural Network (ANN) using **TensorFlow's Keras API**. We'll tackle a classic problem: classifying handwritten digits from the MNIST dataset.

### What is an Artificial Neural Network (ANN)?

An Artificial Neural Network is a computational model inspired by the structure and function of biological neural networks (the human brain). It consists of interconnected "neurons" (nodes) organized in layers, processing information from an input layer, through one or more hidden layers, to an output layer.

### The MNIST Dataset

The MNIST (Modified National Institute of Standards and Technology) dataset is a large database of handwritten digits that is commonly used for training various image processing systems. It consists of 60,000 training images and 10,000 testing images. Each image is a 28x28 pixel grayscale image.

### The Workflow: Building a Simple ANN

#### 1. Loading and Normalizing the Data

First, we load the MNIST dataset and normalize the pixel values from `[0, 255]` to `[0, 1]`. Normalization helps the neural network learn more efficiently.

```python
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1) # Normalizes each image individually
x_test = tf.keras.utils.normalize(x_test, axis=1)
```

#### 2. Building the Neural Network Architecture

We'll create a simple Sequential model, which is a linear stack of layers.

*   **`tf.keras.layers.Flatten`:** This layer reshapes our 28x28 2D image arrays into 1D vectors (28*28 = 784 pixels). This is necessary because the `Dense` layers expect 1D input.
*   **`tf.keras.layers.Dense`:** These are fully connected layers, meaning each neuron in a layer is connected to every neuron in the previous layer.
    *   The first two `Dense` layers are "hidden layers" with 128 neurons each and use the `relu` (Rectified Linear Unit) activation function. `relu` is a common choice for hidden layers.
    *   The final `Dense` layer is the "output layer" with 10 neurons (one for each digit from 0-9) and uses the `softmax` activation function. `softmax` converts the output into a probability distribution, where the sum of all probabilities is 1.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # Input layer
    tf.keras.layers.Dense(128, activation='relu'), # Hidden layer 1
    tf.keras.layers.Dense(128, activation='relu'), # Hidden layer 2
    tf.keras.layers.Dense(10, activation='softmax') # Output layer
])
```

#### 3. Compiling the Model

Compiling the model configures the learning process. We specify:

*   **`optimizer='adam'`:** An efficient algorithm for gradient descent optimization.
*   **`loss='sparse_categorical_crossentropy'`:** This is the appropriate loss function for multi-class classification when the labels are integers (e.g., 0, 1, 2...).
*   **`metrics=['accuracy']`:** We want to monitor the accuracy during training.

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

#### 4. Training the Model

Finally, we train the model using our training data. The `model.fit()` method does the actual learning.

```python
model.fit(x_train, y_train, epochs=3) # Train for 3 epochs
```
After training, you can then evaluate your model's performance on the unseen test data.

### Key Takeaways for Day 20

*   You've built and trained your first Artificial Neural Network!
*   **Normalization** is a crucial preprocessing step for neural networks.
*   **`tf.keras.layers.Flatten`** is used to convert multi-dimensional inputs into a 1D vector.
*   **`tf.keras.layers.Dense`** layers are the workhorses of ANNs.
*   **Activation functions** like `relu` (for hidden layers) and `softmax` (for multi-class output) introduce non-linearity and convert outputs to probabilities, respectively.
*   **`model.compile()`** configures the training process.
*   **`model.fit()`** trains the model.

This is a fundamental step into the world of deep learning. Tomorrow, we'll likely build upon this foundation!