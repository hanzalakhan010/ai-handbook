---
id: day-27
title: 'Day 27: Taking Control of Training with Keras Callbacks'
---

## Day 27: Taking Control of Your Training with Keras Callbacks

### A Quick Recap

Over the last few days, we've learned how to build powerful models using **Transfer Learning** (Day 25) and improve their robustness with **Data Augmentation** (Day 26).

Now that we are training more complex models, the training process itself can become long and difficult to manage. How do you save the best version of your model? How do you stop training if the model isn't improving? This is where **Keras Callbacks** come to the rescue.

### What are Keras Callbacks?

Callbacks are tools that you can use to monitor and influence the training process of your Keras model. You can think of them as little helpers that watch your model as it trains and can perform actions at different stages, like at the beginning or end of an epoch, or before or after a training batch.

### The Most Important Callbacks You Should Know

Here are four of the most useful and commonly used callbacks:

1.  **`ModelCheckpoint`**: This callback saves your model during training. You can configure it to save the model after every epoch, or only when it sees the best performance on a monitored metric (like `val_accuracy`). This is incredibly useful because it means you don't lose all your work if your training is interrupted, and you can easily get the best version of your model at the end of training.

2.  **`EarlyStopping`**: This callback stops the training process automatically when a monitored metric has stopped improving. For example, if your validation loss doesn't improve for a certain number of epochs (the "patience"), `EarlyStopping` will halt the training. This is a fantastic tool to prevent overfitting and save time and resources.

3.  **`ReduceLROnPlateau`**: This callback reduces the learning rate when a metric has stopped improving. The idea is that if your model is stuck in a plateau, reducing the learning rate can help it find a better minimum.

4.  **`TensorBoard`**: This callback logs various metrics during training, which you can then visualize using a tool called TensorBoard. This is great for debugging and comparing different model architectures and training runs.

### How to Use Callbacks

Using callbacks is straightforward. You instantiate the callbacks you want to use and then pass them as a list to the `callbacks` argument in the `model.fit()` method.

### Code Example: Using `ModelCheckpoint` and `EarlyStopping`

Here's how you would use `ModelCheckpoint` and `EarlyStopping` when training a model:

```python
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ... (your model definition here)
# model = ...
# model.compile(...)

# 1. Instantiate the callbacks
# ModelCheckpoint to save the best model
model_checkpoint = ModelCheckpoint(
    filepath='best_model.keras', # File path to save the model
    monitor='val_loss',         # Metric to monitor
    save_best_only=True,        # Only save the best model
    mode='min',                 # 'min' for loss, 'max' for accuracy
    verbose=1
)

# EarlyStopping to stop training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,                 # Number of epochs with no improvement to wait
    mode='min',
    verbose=1,
    restore_best_weights=True   # Restore model weights from the epoch with the best value
)

# 2. Pass the callbacks to model.fit()
# history = model.fit(
#     train_dataset,
#     validation_data=validation_dataset,
#     epochs=100, # Train for a large number of epochs
#     callbacks=[model_checkpoint, early_stopping] # Pass the list of callbacks here
# )

# After training, you can load the best model like this:
# best_model = tf.keras.models.load_model('best_model.keras')
```
With this setup, the training will automatically stop if the validation loss doesn't improve for 5 epochs, and you'll have the best model saved in `best_model.keras`.

---

### <mark>Things I Didn't Go Through (But You Should Explore)</mark>

*   **Creating Custom Callbacks:** You can create your own callbacks to perform specific actions during training. You do this by creating a class that inherits from `tf.keras.callbacks.Callback` and then implementing methods like `on_epoch_end`, `on_batch_begin`, etc. This gives you ultimate control over the training loop.
*   **`LearningRateScheduler`:** This is another callback for controlling the learning rate. It allows you to define a custom function that takes the current epoch and learning rate and returns a new learning rate. This is useful for implementing more complex learning rate schedules (e.g., "learning rate warm-up").
*   **`CSVLogger`:** A simple callback that streams epoch results to a CSV file. This can be an easy way to keep a log of your training history.

## Small Project: Training with a Full Set of Callbacks

**Objective:** Use `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau` together to manage a training process efficiently and effectively.

**Dataset:** We'll use the "Cats vs. Dogs" dataset again, but this time we will train a model from scratch, which is more prone to overfitting and can benefit more from callbacks.

**Steps:**

1.  **Load and Prepare Data:**
    *   Load the "cats_vs_dogs" dataset using `tensorflow_datasets` as you did in Day 22. Prepare your training and validation batches.

2.  **Build a Model:**
    *   Build a simple CNN from scratch. It doesn't have to be perfect. A few `Conv2D` and `MaxPooling2D` layers followed by a `Dense` head will work. The goal is to have a model that can overfit or get stuck, giving our callbacks a chance to work.

3.  **Define Your Callbacks:**
    *   **`ModelCheckpoint`:** Configure it to save the best model to a file named `best_cat_dog_model.keras`. Monitor `val_accuracy` and set `save_best_only=True`.
    *   **`EarlyStopping`:** Configure it to monitor `val_loss` with a `patience` of 5. This will stop the training if the validation loss doesn't improve for 5 epochs.
    *   **`ReduceLROnPlateau`:** Configure this to monitor `val_loss`. If the loss plateaus for, say, 3 epochs (`patience=3`), it will reduce the learning rate.

4.  **Compile and Train:**
    *   Compile your model.
    *   Start the training by calling `model.fit()`. Set the number of epochs to a large number, like `50`. Don't worry, `EarlyStopping` will likely stop it before then.
    *   Pass your list of three callbacks to the `callbacks` argument.

5.  **Analyze the Output:**
    *   Watch the console output during training. You should see messages from the callbacks:
        *   `ModelCheckpoint` will print a message every time it saves a new best model.
        *   `ReduceLROnPlateau` will print a message when it reduces the learning rate.
        *   `EarlyStopping` will print a message when it stops the training.
    *   After training stops, how many epochs did it actually run? Did the learning rate get reduced?

**Key Takeaway:** This project teaches you how to use a suite of callbacks to automate the most tedious parts of model training. You'll learn to set up a "fire-and-forget" training job that automatically saves the best model, stops when it's done improving, and tries to navigate tricky parts of the loss landscape by adjusting the learning rate.

Happy training!
