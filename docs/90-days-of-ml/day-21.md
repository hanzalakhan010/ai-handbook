---
id: day-21
title: 'Day 21: Introduction to Convolutional Neural Networks (CNNs)'
---

## Day 21: Introduction to Convolutional Neural Networks (CNNs)

Yesterday, we built our first Artificial Neural Network (ANN) to classify handwritten digits. While ANNs can work for images, they are not ideal. Today, we introduce a specialized type of neural network designed specifically for processing visual data: **Convolutional Neural Networks (CNNs)**.

### Why Not Just Use ANNs for Images?

*   **High Dimensionality:** Images have many pixels (e.g., a 100x100 RGB image has 100\*100\*3 = 30,000 features). A fully connected ANN with just one hidden layer of 100 neurons would have 30,000\*100 = 3 million weights, making it computationally expensive and prone to overfitting.
*   **Loss of Spatial Information:** ANNs treat image pixels as a flat vector, losing valuable spatial relationships between nearby pixels (e.g., a pixel's value is highly correlated with its neighbors).
*   **Lack of Translation Invariance:** ANNs are not inherently good at recognizing patterns if they appear in different locations in the image.

### What is a Convolutional Neural Network (CNN)?

CNNs are designed to address these limitations by leveraging the spatial structure of images. They consist of specialized layers that automatically learn relevant features from the input data.

The core components of a CNN are:

1.  **Convolutional Layer:**
    *   **Filters (Kernels):** Small matrices that slide (convolve) over the input image, performing dot products with the underlying pixels. Each filter learns to detect a specific feature (e.g., edges, textures, corners).
    *   **Feature Maps:** The output of a convolutional layer, showing where the filter detected its feature in the input.
    *   **Local Receptive Fields:** Each neuron in a convolutional layer is connected only to a small, local region of the input.
    *   **Weight Sharing:** The same filter is applied across the entire image, drastically reducing the number of parameters and enabling translation invariance.

2.  **Pooling Layer (Downsampling):**
    *   These layers reduce the spatial dimensions (width and height) of the feature maps, reducing the amount of computation and memory.
    *   **Max Pooling:** The most common type, it takes the maximum value from a patch of the feature map. This helps make the network more robust to small shifts in the input.

3.  **Activation Functions:**
    *   Typically, ReLU (`tf.keras.layers.ReLU`) is used after convolutional layers to introduce non-linearity.

4.  **Fully Connected Layer (Dense Layer):**
    *   After several convolutional and pooling layers, the learned feature maps are flattened into a 1D vector and fed into one or more fully connected (Dense) layers, just like in a traditional ANN, for classification.

### Architecture of a Simple CNN

A typical CNN architecture looks like this:

`Input Image -> Conv Layer -> ReLU -> Pooling Layer -> Conv Layer -> ReLU -> Pooling Layer -> Flatten -> Dense Layer -> ReLU -> Output (Dense) Layer (Softmax)`

### Code Example: Building a Simple CNN for Image Classification

Let's build a simple CNN using Keras, similar to how the notebook imports suggest it might be processing image data. We'll use the MNIST dataset again for simplicity, showing how a CNN can achieve higher accuracy.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# CNNs expect a 4D input (batch_size, height, width, channels).
# For grayscale images like MNIST, channels = 1.
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")
```

### Key Takeaways for Day 21

*   **CNNs** are specialized neural networks for image processing.
*   **Convolutional layers** learn features through filters and weight sharing.
*   **Pooling layers** (e.g., Max Pooling) reduce dimensionality and introduce translation invariance.
*   CNNs combine these layers with traditional **Fully Connected layers** for classification.

This foundation is crucial for almost any task involving image data. Tomorrow, we might explore further aspects of CNNs or another deep learning architecture!

---

### <mark>Things I Didn't Go Through (But You Should Explore)</mark>

*   **Strides and Padding:** How the filter moves across the image (stride) and how borders are handled (padding).
*   **Different Pooling Types:** Average Pooling as an alternative to Max Pooling.
*   **Advanced Architectures:** Research famous CNN architectures like LeNet, AlexNet, VGG, ResNet, and Inception.
*   **Data Augmentation:** For real-world image datasets, data augmentation (covered in Day 26) is critical to prevent overfitting and improve generalization. You would integrate augmentation layers at the beginning of your CNN model.

## Small Project: CIFAR-10 Image Classification

**Objective:** Build a CNN from scratch to classify images from the CIFAR-10 dataset, which is more challenging than MNIST.

**Dataset:** The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) contains 60,000 32x32 color images in 10 classes (e.g., airplane, dog, cat, horse). It's available directly in `tf.keras.datasets`.

**Steps:**

1.  **Load and Prepare the Data:**
    *   `from tensorflow.keras.datasets import cifar10`
    *   ` (x_train, y_train), (x_test, y_test) = cifar10.load_data()`
    *   Normalize the pixel values by dividing by 255.0.
    *   The labels `y_train` and `y_test` are arrays of integers. For a multi-class classification model with a `softmax` output, you should one-hot encode them using `tf.keras.utils.to_categorical`.

2.  **Build a Deeper CNN:**
    *   Your MNIST model was a good start. For CIFAR-10, you'll likely need a deeper model.
    *   Try building a model with the following structure:
        *   `Conv2D` (32 filters, 3x3 kernel, `relu` activation, `padding='same'`) -> `Conv2D` (32 filters...) -> `MaxPooling2D`
        *   `Conv2D` (64 filters, 3x3 kernel, `relu` activation, `padding='same'`) -> `Conv2D` (64 filters...) -> `MaxPooling2D`
        *   `Flatten`
        *   `Dense` (512 units, `relu` activation)
        *   `Dropout(0.5)` (To help prevent overfitting)
        *   `Dense` (10 units, `softmax` activation) - for the 10 classes.
    *   The `padding='same'` argument ensures that the output feature map has the same height and width as the input.

3.  **Compile and Train the Model:**
    *   Compile the model using the `adam` optimizer and `categorical_crossentropy` loss (because we one-hot encoded the labels).
    *   Train the model for a reasonable number of epochs (e.g., 20-30). Use the `validation_data` argument in `model.fit()` to monitor its performance on the test set.

4.  **Evaluate and Visualize:**
    *   Evaluate the final model on the test set. What accuracy do you achieve? (Don't be discouraged if it's not perfect; CIFAR-10 is a non-trivial problem!).
    *   Plot the training and validation accuracy/loss curves to check for overfitting.

**Key Takeaway:** This project will give you experience in building a more realistic CNN for a standard computer vision benchmark. You will learn how to handle multi-class labels with one-hot encoding and how to stack convolutional layers to build a deeper, more powerful model.
