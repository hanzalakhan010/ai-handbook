---
id: day-25
title: 'Day 25: Transfer Learning for Medical Imaging'
---

## Day 25: Transfer Learning for Medical Imaging

### A Quick Recap

In Day 24, we discussed the importance of model performance metrics like **precision** and **recall**, especially in the context of our pneumonia detection model. We also touched upon deploying models on edge devices using **TensorFlow Lite**.

While building custom models is a great learning experience, achieving high accuracy, particularly with limited data, can be challenging. This is where **Transfer Learning** comes in.

### What is Transfer Learning?

Transfer Learning is a machine learning technique where a model developed for a task is reused as the starting point for a model on a second task.

In computer vision, we can use a model trained on a very large dataset (like ImageNet, which has millions of images) and adapt it to our specific task, like detecting pneumonia in X-ray images. The knowledge (features, patterns) the model learned from the large dataset is "transferred" to our new task.

### Why Use Transfer Learning?

1.  **Less Data Required:** You can achieve good results even with a smaller dataset because the model has already learned to recognize general features from the large dataset.
2.  **Faster Training:** Since the base model is already trained, you only need to train the new parts of the model and/or fine-tune the existing layers. This saves a lot of time and computational resources.
3.  **Better Performance:** Pre-trained models are often very deep and complex, and have been trained by experts on massive datasets. This usually leads to better performance than a model trained from scratch on a small dataset.

### How to Use Transfer Learning: A Step-by-Step Guide

Here’s a general workflow for using transfer learning for an image classification task:

1.  **Choose a Pre-trained Model:** Select a well-known model architecture that suits your needs. Some popular choices include VGG16, ResNet50, InceptionV3, and MobileNetV2. MobileNetV2 is a good choice for mobile and edge devices due to its small size and efficiency.
2.  **Instantiate the Base Model:** Load the pre-trained model without its final classification layer (the "head"). We'll add our own custom head.
3.  **Freeze the Base Model:** "Freeze" the layers of the base model. This prevents their weights from being updated during training. We want to keep the learned features from the original dataset.
4.  **Add a New Classification Head:** Add your own layers on top of the frozen base model. This new "head" will be a small neural network that takes the output of the base model and learns to classify your specific images (e.g., 'pneumonia' vs. 'normal').
5.  **Train the New Head:** Train your model. Only the weights of the new classification head will be updated.
6.  **(Optional but Recommended) Fine-Tuning:** After the new head has been trained, you can "unfreeze" some of the top layers of the base model and continue training with a very low learning rate. This allows the model to adapt its learned features more specifically to your dataset.

### Code Example: Transfer Learning with Keras

Here's a simplified example of how to implement transfer learning using TensorFlow/Keras with the VGG16 model.

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# 1. Instantiate the base model (VGG16)
# We are not including the top (classification) layer.
# The input shape should match your images.
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 2. Freeze the base model
base_model.trainable = False

# 3. Add a new classification head
# We will add a GlobalAveragePooling2D layer to reduce the dimensions,
# and then a Dense layer for our final classification.
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid') # For binary classification (e.g., pneumonia vs. normal)
])

# 4. Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])

model.summary()

# Now you can train this model on your dataset of X-ray images.
# history = model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

# 5. (Later) Fine-tuning
# After initial training, you can unfreeze some layers and train again.
# base_model.trainable = True
# for layer in base_model.layers[:-4]: # Freeze all but the top 4 layers
#     layer.trainable = False

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Use a very low learning rate
#               loss='binary_crossentropy',
#               metrics=['accuracy', 'Precision', 'Recall'])

# history_fine_tune = model.fit(train_dataset, validation_data=validation_dataset, epochs=10)
```

---

### <mark>Things I Didn't Go Through (But You Should Explore)</mark>

This is a new and important topic, and there's a lot more to learn. Here are some things you should look into next:

*   **Choosing the Right Pre-trained Model:**
    *   **VGG16/19:** Good performance, but large and slow.
    *   **ResNet:** Very popular, good performance, and deeper than VGG.
    *   **Inception:** Complex, but very efficient and high-performing.
    *   **MobileNet:** Lightweight, fast, and designed for mobile/edge devices. This could be a great choice for your pneumonia detector if you want to deploy it on a mobile device.
    *   **EfficientNet:** A newer family of models that achieve state-of-the-art accuracy with smaller size and faster inference.
*   **Fine-Tuning Strategies:**
    *   How many layers should you unfreeze? There's no single answer. It depends on your dataset size and similarity to the original dataset (ImageNet). A common approach is to unfreeze the top block of layers first and see if it improves performance.
*   **Data Augmentation:**
    *   To get the most out of your model, you should use data augmentation. This involves creating modified versions of your training images (e.g., rotating, zooming, flipping). This helps the model generalize better and reduces overfitting. Keras has `ImageDataGenerator` and layers like `tf.keras.layers.RandomFlip` that make this easy.

## Small Project: Fine-Tuning for Pneumonia Detection

**Objective:** Implement a full transfer learning workflow, including **fine-tuning**, to build a high-performance classifier for the Pneumonia X-Ray dataset.

**Dataset:** The [Pneumonia X-Ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) on Kaggle. You'll need to set up `train`, `val`, and `test` directories and use `tf.keras.utils.image_dataset_from_directory` to load the data.

**Steps:**

1.  **Load and Prepare Data:**
    *   Use `image_dataset_from_directory` to create training and validation datasets.
    *   Use data augmentation layers (`tf.keras.layers.RandomFlip`, `tf.keras.layers.RandomRotation`) to create a data augmentation pipeline. Apply this to your training dataset.
    *   Use `tf.keras.applications.mobilenet_v2.preprocess_input` to preprocess your data as expected by the MobileNetV2 model.

2.  **Phase 1: Feature Extraction:**
    *   Load the `MobileNetV2` model as your base model, with `include_top=False`.
    *   Freeze the base model: `base_model.trainable = False`.
    *   Add your custom classification head (`GlobalAveragePooling2D`, `Dropout`, `Dense`).
    *   Compile and train this model for a few epochs (e.g., 5-10) until the validation accuracy plateaus. This warms up your classification head.

3.  **Phase 2: Fine-Tuning:**
    *   Unfreeze the base model: `base_model.trainable = True`.
    *   It's a good practice to only unfreeze the top layers. Let's unfreeze the last 20 layers:
        *   `fine_tune_at = 100`
        *   `for layer in base_model.layers[:fine_tune_at]: layer.trainable = False`
    *   **Re-compile the model** with a **very low learning rate**. This is critical to prevent a large gradient update from destroying the pre-trained weights. Use something like `Adam(learning_rate=0.00001)`.
    *   Continue training the model (`model.fit(...)`) for more epochs. You can start this training from where the previous phase left off by passing the `initial_epoch` argument.

4.  **Evaluate:**
    *   Evaluate your final, fine-tuned model on the test set.
    *   Plot the accuracy and loss curves for both the initial training and the fine-tuning phases to see the effect of fine-tuning. You should see a second jump in performance after you start fine-tuning.

**Key Takeaway:** This project takes you through the complete, state-of-the-art process for transfer learning. You will learn the critical difference between feature extraction and fine-tuning and understand the importance of a low learning rate when updating pre-trained weights. This two-phase approach is fundamental to achieving high performance in modern computer vision tasks.
