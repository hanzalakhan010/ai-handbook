---
id: day-26
title: 'Day 26: Data Augmentation'
---

## Day 26: Data Augmentation to Boost Model Performance

### A Quick Recap

In Day 25, we explored **Transfer Learning**, a powerful technique to build accurate models by reusing pre-trained architectures. We learned that even with transfer learning, the performance of our model heavily depends on the quality and quantity of our own data.

But what if you don't have a massive dataset? That's where **Data Augmentation** comes in.

### What is Data Augmentation?

Data Augmentation is a technique used to artificially increase the size of a training dataset by creating modified versions of its images.

The goal is to expose our model to a wider variety of training examples, which helps it learn the underlying patterns better and become more robust. This, in turn, improves the model's ability to generalize to new, unseen images and significantly reduces overfitting.

### Common Data Augmentation Techniques

Here are some of the most common augmentation techniques used in computer vision:

*   **Flipping:** Creating mirror images, either horizontally or vertically.
*   **Rotation:** Rotating the image by a certain angle.
*   **Zooming:** Zooming in or out on the image.
*   **Cropping:** Randomly cropping a part of the image.
*   **Brightness/Contrast:** Adjusting the brightness or contrast of the image.
*   **Adding Noise:** Adding random noise to the image.

### How to Implement Data Augmentation in Keras/TensorFlow

There are two main ways to perform data augmentation in a TensorFlow/Keras pipeline:

1.  **Using `ImageDataGenerator`:** This is the traditional way. You configure the augmentation parameters on an `ImageDataGenerator` instance and then use it to load and augment your data from a directory.
2.  **Using Keras Preprocessing Layers:** This is the more modern and recommended approach. You add augmentation layers directly into your model definition. These layers are only active during training and are inactive during inference. A major advantage is that the augmentation happens on the GPU, which is much faster.

### Code Example: Using Keras Preprocessing Layers

Here's how you can add data augmentation layers to your model. This is the recommended way to do it.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 224

# Define the data augmentation layers
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# Now, you can include the data_augmentation model as the first layer of your main model.
# Let's use the transfer learning model from yesterday as an example.
from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = models.Sequential([
    data_augmentation, # Apply augmentation as the first layer
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# When you call model.fit(), the data augmentation layers will be active.
# When you call model.predict() or model.evaluate(), they will be inactive.
# history = model.fit(train_dataset, validation_data=validation_dataset, epochs=10)
```

**Important Note:** Data augmentation should only be applied to your **training set**. You should not augment your validation or test sets, as they are supposed to represent the real-world, unseen data. The Keras preprocessing layers handle this automatically.

---

### <mark>Things I Didn't Go Through (But You Should Explore)</mark>

*   **Advanced Data Augmentation Techniques:**
    *   **CutOut/Random Erasing:** Randomly erasing a rectangular region in an image. This forces the model to learn from the entire context of the image, not just the most prominent features.
    *   **MixUp:** Creating new images by taking a weighted average of two images and their labels.
    *   **CutMix:** A variation of CutOut where the erased region is replaced with a patch from another image.
*   **Albumentations Library:** For more advanced and highly optimized augmentation pipelines, check out the `albumentations` library. It's a very popular, open-source library that offers a wide variety of augmentation techniques and is often faster than TensorFlow's built-in methods.
*   **The effect of different augmentations:** Experiment with different combinations of augmentations to see how they affect your model's performance. The best set of augmentations can be problem-dependent.

## Small Project: The Impact of Augmentation on Overfitting

**Objective:** Visually and empirically understand how data augmentation helps prevent overfitting by training two models—one with augmentation and one without—and comparing their training curves.

**Dataset:** The [TensorFlow Flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers) is a great dataset for this task.

**Steps:**

1.  **Load and Prepare the Data:**
    *   Use `tf.keras.utils.image_dataset_from_directory` to load the flower dataset. Create training and validation splits.
    *   Make sure to configure the datasets for performance with `.cache()` and `.prefetch()`.

2.  **Define Your Augmentation Layers:**
    *   Create a `Sequential` model for data augmentation, including layers like `RandomFlip`, `RandomRotation`, and `RandomZoom`.

3.  **Build and Train Model #1: WITHOUT Augmentation:**
    *   Build a simple CNN. You can use a simplified version of the transfer learning model from previous days (e.g., a frozen `MobileNetV2` base and a classification head) or a small CNN from scratch.
    *   Compile the model.
    *   Train the model for a significant number of epochs (e.g., 30-40) and store its `history`.

4.  **Build and Train Model #2: WITH Augmentation:**
    *   Build the exact same model architecture as before.
    *   **Important:** Add your data augmentation model as the very first layer.
    *   Compile the model (using the same optimizer and loss).
    *   Train the model for the same number of epochs and store its `history`.

5.  **Visualize and Compare:**
    *   Create a plot that shows the training and validation accuracy curves for **both models** on the same axes.
    *   Create a second plot that shows the training and validation loss curves for both models.
    *   What do you observe?
        *   The "without augmentation" model should show a large gap between its training accuracy (which will be very high) and its validation accuracy (which will plateau or drop). This is classic overfitting.
        *   The "with augmentation" model should have training and validation curves that are much closer together, indicating better generalization.

**Key Takeaway:** This project provides a clear, visual demonstration of the power of data augmentation. You will see firsthand how it acts as a form of regularization, helping your model learn more robust features and perform better on unseen data.

Happy augmenting!
