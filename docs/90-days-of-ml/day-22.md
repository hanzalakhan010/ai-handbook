---
id: day-22
title: 'Day 22: Transfer Learning with Pre-trained CNNs'
---

## Day 22: Transfer Learning with Pre-trained CNNs

Yesterday, we built our first Convolutional Neural Network (CNN) from scratch. While powerful, training a robust CNN, especially for complex tasks and large datasets, often requires a significant amount of labeled data and computational resources.

This is where **Transfer Learning** becomes invaluable, particularly in domains like medical imaging where labeled data can be scarce.

### Reintroducing Transfer Learning (Now for CNNs!)

We briefly touched upon Transfer Learning in Day 25, where the general concept was introduced. Today, we apply this powerful paradigm specifically to Convolutional Neural Networks.

**Transfer Learning for CNNs** means taking a pre-trained CNN (a model that has already been trained on a very large and diverse image dataset, like ImageNet) and reusing its learned features as a starting point for a new, similar image classification task.

### Why Transfer Learning for CNNs?

1.  **Leverage Pre-trained Knowledge:** Large datasets like ImageNet (millions of images, 1000 categories) have allowed models to learn a rich hierarchy of features, from basic edges and textures in early layers to complex object parts in deeper layers. This learned knowledge is highly transferable.
2.  **Reduced Data Requirements:** You don't need a huge dataset for your specific task. The pre-trained model has already learned general visual patterns.
3.  **Faster Training:** Training only a small part of the model (or fine-tuning) is significantly faster than training a deep CNN from scratch.
4.  **Better Performance:** Often, models using transfer learning outperform models trained from scratch on smaller, specialized datasets.

### Common Pre-trained Architectures

Many powerful CNN architectures are available as pre-trained models:

*   **VGG16/19:** Known for their simplicity and depth.
*   **ResNet (e.g., ResNet50):** Introduced "residual connections" to enable training much deeper networks.
*   **Inception (e.g., InceptionV3):** Uses "inception modules" for efficient computation.
*   **MobileNet (e.g., MobileNetV2):** Designed to be lightweight and efficient, ideal for mobile and edge devices.

### Two Main Strategies for Transfer Learning with CNNs

1.  **Feature Extraction:**
    *   You use the pre-trained CNN as a fixed feature extractor.
    *   You remove the original classification head of the pre-trained model.
    *   You add your own new classification layers (Dense layers) on top.
    *   Only these new layers are trained; the weights of the pre-trained base model remain frozen.
    *   This is typically used when your new dataset is small and similar to the original dataset the pre-trained model was trained on.

2.  **Fine-tuning:**
    *   Similar to feature extraction, you remove the original classification head and add your own.
    *   However, after initial training of the new layers, you unfreeze some (usually the top-most) layers of the pre-trained base model.
    *   The entire model (frozen base + unfrozen base layers + new head) is then trained with a very low learning rate.
    *   This allows the pre-trained features to be slightly adjusted ("fine-tuned") to be more specific to your new dataset.
    *   This is typically used when your new dataset is larger and/or more dissimilar to the original training data.

### Code Example: Feature Extraction with MobileNetV2

Here's an example of using **MobileNetV2** as a feature extractor in TensorFlow/Keras.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import numpy as np # For dummy data generation

# --- 1. Load Pre-trained Base Model (Feature Extractor) ---
# include_top=False: exclude the ImageNet classification head
# input_shape: must match the expected input for MobileNetV2 (e.g., 224x224 RGB)
base_model = MobileNetV2(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')

# Freeze the base model layers
base_model.trainable = False

# --- 2. Build Your Custom Classification Head ---
# We'll add a Global Average Pooling layer and a Dense layer for classification.
global_average_layer = layers.GlobalAveragePooling2D()
prediction_layer = layers.Dense(1, activation='sigmoid') # For binary classification (e.g., cat vs dog)

# --- 3. Connect the Base Model to the Head ---
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False) # Important: base_model in inference mode
x = global_average_layer(x)
outputs = prediction_layer(x)
model = models.Model(inputs, outputs)

# --- 4. Compile the Model ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 5. Train the Model (Conceptual with dummy data) ---
# Replace this with your actual image data generator or tf.data pipeline
# Dummy data for demonstration
dummy_train_images = np.random.rand(100, 224, 224, 3)
dummy_train_labels = np.random.randint(0, 2, 100)

# history = model.fit(
#     dummy_train_images, dummy_train_labels,
#     epochs=10,
#     validation_data=(dummy_val_images, dummy_val_labels) # Use your validation data
# )
```

### Key Takeaways for Day 22

*   **Transfer Learning** is essential for deep learning, especially with limited data.
*   **Pre-trained CNNs** offer a powerful starting point for image tasks.
*   **Feature Extraction** involves freezing the base model and training a new classification head.
*   **Fine-tuning** extends this by unfreezing and retraining top layers of the base model.

This technique is fundamental to achieving state-of-the-art results in computer vision without needing to train massive models from scratch.

---

### <mark>Things I Didn't Go Through (But You Should Explore)</mark>

*   **Data Augmentation:** Always use data augmentation (covered in Day 26) when working with image data and transfer learning. It helps prevent overfitting on your smaller dataset.
*   **Choosing the Right Pre-trained Model:** Research different pre-trained models (VGG, ResNet, Inception, MobileNet) and understand their trade-offs in terms of accuracy, size, and computational cost for your specific application.
*   **Fine-tuning Details:** Experiment with which layers to unfreeze, and use very low learning rates for fine-tuning to avoid destroying the pre-trained weights.
*   **Learning Rate Schedulers:** Techniques to dynamically adjust the learning rate during fine-tuning (e.g., cosine decay).

## Small Project: Cat vs. Dog Classification

**Objective:** Use transfer learning with a pre-trained model to build a classifier that can distinguish between images of cats and dogs.

**Dataset:** The [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) from Kaggle is a classic computer vision problem. You can download a subset from the TensorFlow website directly.

**Steps:**

1.  **Download and Prepare the Dataset:**
    *   TensorFlow makes it easy to download and prepare this dataset.
    *   ```python
        import tensorflow_datasets as tfds
        
        # Load the dataset
        (train_ds, val_ds, test_ds), metadata = tfds.load(
            'cats_vs_dogs',
            split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
            with_info=True,
            as_supervised=True,
        )
        
        # You'll also need a function to resize images to the expected input of your base model
        IMG_SIZE = 160 # For MobileNetV2
        def format_example(image, label):
            image = tf.cast(image, tf.float32)
            image = (image/127.5) - 1 # Normalize to [-1, 1] as expected by MobileNetV2
            image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
            return image, label
            
        train_batches = train_ds.map(format_example).batch(32)
        validation_batches = val_ds.map(format_example).batch(32)
        ```

2.  **Build the Transfer Learning Model (Feature Extraction):**
    *   Use `MobileNetV2` as your base model, as it's efficient. Load it with `weights='imagenet'` and `include_top=False`.
    *   Set `base_model.trainable = False`.
    *   Add a `GlobalAveragePooling2D` layer and a final `Dense(1, activation='sigmoid')` layer for your binary classification head.
    *   Combine them into a `Model` as shown in the lesson.

3.  **Compile and Train the Head:**
    *   Compile the model with a suitable optimizer (`Adam` is a good choice) and `binary_crossentropy` loss.
    *   Train the model for a few epochs using `model.fit(train_batches, validation_data=validation_batches, epochs=5)`. You should see the validation accuracy improve quickly.

4.  **Evaluate Your Model:**
    *   Evaluate the model on the `test_ds` (make sure to apply the same `format_example` and batching).
    *   What accuracy do you get? It should be quite high, even with only a few epochs of training on the new head.

**Key Takeaway:** This project provides a complete, hands-on workflow for solving a real-world image classification problem using transfer learning. You'll learn how to prepare a dataset using `tf.data` and see just how effective the feature extraction strategy can be.
