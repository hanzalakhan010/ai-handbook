---
id: day-28
title: 'Day 28: A Complete Workflow for Training and Evaluation'
---

## Day 28: A Complete Workflow for Training and Evaluation

### A Quick Recap

In the last few days, we've gathered a powerful set of tools for training deep learning models:
*   **Transfer Learning** (Day 25) to build upon the knowledge of pre-trained models.
*   **Data Augmentation** (Day 26) to make our models more robust.
*   **Keras Callbacks** (Day 27) to monitor and control the training process.

Now, let's put it all together and outline a complete, end-to-end workflow for training and evaluating a computer vision model.

### The Big Picture: A 7-Step Workflow

Here is a step-by-step guide that you can follow for your own projects.

#### 1. Data Preparation

This is the first and one of the most crucial steps.
*   **Load Your Data:** Load your images and labels from their directories.
*   **Create Datasets:** Create `tf.data.Dataset` objects for your training, validation, and test sets. Using `tf.data` is highly recommended as it provides a very efficient way to build data pipelines.
*   **Prefetching and Caching:** Use `.cache()` and `.prefetch()` on your datasets to speed up the data loading process.

#### 2. Model Building

*   **Data Augmentation:** Define your data augmentation layers using `tf.keras.layers.experimental.preprocessing`.
*   **Base Model:** Instantiate your pre-trained base model (e.g., MobileNetV2, ResNet50) using Transfer Learning. Remember to freeze its weights.
*   **Connect the Pieces:** Create a `tf.keras.Sequential` model that starts with the augmentation layers, followed by the base model, and ends with your custom classification head.

#### 3. Model Compilation

*   **Choose an Optimizer:** `tf.keras.optimizers.Adam` is usually a good default choice.
*   **Choose a Loss Function:** For binary classification (like 'pneumonia' vs. 'normal'), use `'binary_crossentropy'`. For multi-class classification, use `'categorical_crossentropy'`.
*   **Choose Metrics:** `'accuracy'` is a good starting point. You can also include `tf.keras.metrics.Precision()` and `tf.keras.metrics.Recall()`.

#### 4. Callbacks Definition

*   **`ModelCheckpoint`**: To save the best version of your model during training.
*   **`EarlyStopping`**: To prevent overfitting and save time.

#### 5. Model Training

*   **Call `model.fit()`:** Pass your training and validation datasets, the number of epochs, and your list of callbacks to the `fit` method.
*   **Store the History:** The `fit` method returns a `history` object that contains the training and validation metrics for each epoch.

#### 6. Model Evaluation

*   **Load the Best Model:** Load the best model that was saved by `ModelCheckpoint`.
*   **Evaluate on Test Set:** Evaluate the model on your test set (which the model has never seen before) using `model.evaluate()`. This gives you the final performance metrics.
*   **Visualize Results:** Plot the accuracy and loss curves from the `history` object to see how your model trained.

#### 7. Saving the Final Model

*   **Save for Inference:** Save the final, trained model in the `.keras` format. This model is now ready to be used for predictions.

### Code Example: Putting It All Together

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# --- 1. Data Preparation (Assuming you have train, validation, and test datasets) ---
# train_dataset, validation_dataset, test_dataset = ... (load your tf.data.Dataset objects here)

# --- 2. Model Building ---
IMG_SIZE = 224
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

# --- 3. Model Compilation ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- 4. Callbacks Definition ---
checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_loss")
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# --- 5. Model Training ---
# history = model.fit(train_dataset,
#                     epochs=50,
#                     validation_data=validation_dataset,
#                     callbacks=[checkpoint, early_stopping])

# --- 6. Model Evaluation ---
# loaded_model = tf.keras.models.load_model("best_model.keras")
# loss, accuracy = loaded_model.evaluate(test_dataset)
# print(f"Test Accuracy: {accuracy*100:.2f}%")

# Plotting training history
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()


# --- 7. Saving the Final Model ---
# loaded_model.save("final_pneumonia_detector.keras")

```

---

### <mark>Things I Didn't Go Through (But You Should Explore)</mark>

*   **Hyperparameter Tuning:** We've been using default values for many things (like the learning rate). The process of finding the optimal set of hyperparameters is called **Hyperparameter Tuning**. It can be a complex process, but tools like **KerasTuner** and **Optuna** can help you automate it.
*   **Experiment Tracking:** When you run many experiments with different models and hyperparameters, it can be hard to keep track of what worked and what didn't. Tools like **TensorBoard**, **Weights & Biases**, and **MLflow** are designed to help you log, compare, and visualize your experiments.

## Small Project: End-to-End Intel Image Classifier

**Objective:** Apply the complete 7-step workflow from today's lesson to build, train, and evaluate a robust image classifier for a new dataset.

**Dataset:** The [Intel Image Classification dataset](https://www.kaggle.com/puneet6060/intel-image-classification). This dataset contains images of natural scenes categorized into 6 classes (buildings, forest, glacier, mountain, sea, street).

**Steps (Follow the 7-Step Workflow):**

1.  **Data Preparation:**
    *   Download the dataset and organize it into `train` and `test` directories.
    *   Use `tf.keras.utils.image_dataset_from_directory` to create your training and validation datasets from the `seg_train` directory (use `validation_split=0.2`).
    *   Create your test dataset from the `seg_test` directory.
    *   Configure all datasets for performance with `.cache()` and `.prefetch()`.

2.  **Model Building:**
    *   Define data augmentation layers (`RandomFlip`, `RandomRotation`).
    *   Instantiate a pre-trained base model (`MobileNetV2` is a great choice). Freeze its weights.
    *   Combine the augmentation layers, the base model, and a new classification head. Your head will need a `Dense` layer with **6 units** and a `softmax` activation for the 6 classes.

3.  **Model Compilation:**
    *   Compile the model with an `Adam` optimizer, `sparse_categorical_crossentropy` loss (since the labels from the generator are integers), and `['accuracy']` as the metric.

4.  **Callbacks Definition:**
    *   Define a `ModelCheckpoint` callback to save the best model based on `val_loss`.
    *   Define an `EarlyStopping` callback with a `patience` of 5.

5.  **Model Training:**
    *   Call `model.fit()`, passing in your datasets and callbacks. Train for up to 50 epochs.

6.  **Model Evaluation:**
    *   Load the best model saved by `ModelCheckpoint`.
    *   Evaluate its performance on the test set using `model.evaluate()`. How well did it do?
    *   Plot the training/validation accuracy and loss curves.

7.  **Saving the Final Model:**
    *   Save your final, evaluated model to a file named `intel_image_classifier.keras`.

**Key Takeaway:** This project is a capstone exercise that reinforces the entire end-to-end process of a typical deep learning project. By applying the full workflow to a new dataset, you will solidify your understanding of each step and build confidence in your ability to tackle your own computer vision problems.

This workflow provides a solid foundation for your deep learning projects. Happy coding!
