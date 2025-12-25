---
id: day-29
title: 'Day 29: Saving, Loading, and Converting Your Model'
---

## Day 29: Saving, Loading, and Converting Your Model for Deployment

### A Quick Recap

In Day 28, we established a complete workflow for training and evaluating a model. The final output of that workflow is a trained, high-performing model saved in a file.

But what's next? A model file on your computer isn't very useful in the real world. We need to prepare it for deployment. This involves understanding how to save and load models correctly, and how to convert them into formats suitable for different environments.

### Saving and Loading Keras Models

When you save a Keras model, you are saving everything:
*   The model's architecture
*   The model's weights
*   The training configuration (optimizer, loss, metrics)

There are two main formats for saving a TensorFlow/Keras model:

1.  **The Keras `.keras` format:** This is the modern, recommended format. It's a simple and efficient way to save your model.
2.  **The TensorFlow `SavedModel` format:** This is the more general TensorFlow format. It's saved as a directory containing several files.

For most use cases, the `.keras` format is the easiest to work with.

#### Code Example:

```python
import tensorflow as tf

# Assuming 'model' is your trained Keras model
# model = ...

# --- Saving the model ---
model.save("my_model.keras")

# --- Loading the model ---
# You can now load this model back from the file at any time
loaded_model = tf.keras.models.load_model("my_model.keras")

# The loaded model is already compiled and ready to be used
loaded_model.summary()
```

### From Trained Model to Deployed Model: The Need for Conversion

A model trained on a powerful machine with a GPU is often too large and slow to run on a resource-constrained device like a smartphone or a Raspberry Pi.

This is why we need to **convert** the model. The conversion process optimizes the model for inference, making it smaller and faster.

### Converting to TensorFlow Lite (`.tflite`)

**TensorFlow Lite (TFLite)** is a set of tools that helps developers run their models on mobile, embedded, and IoT devices. It was briefly mentioned in Day 24, but now let's see how to actually create a TFLite model.

The conversion process is straightforward:

1.  Start with a trained Keras or `SavedModel`.
2.  Use the `tf.lite.TFLiteConverter` to perform the conversion.
3.  Save the converted model as a `.tflite` file.

#### Code Example: Converting a Keras Model

```python
import tensorflow as tf

# 1. Load your trained Keras model
model = tf.keras.models.load_model("my_model.keras")

# 2. Create a TFLiteConverter object
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Convert the model
tflite_model = converter.convert()

# 4. Save the converted model to a .tflite file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TensorFlow Lite successfully!")
```
This `model.tflite` file is now ready to be deployed in a mobile app or on an embedded device.

---

### <mark>A Brief Look at Other Deployment Targets</mark>

*   **TensorFlow.js (for web browsers):** If you want to run your model directly in a web browser, you can convert it to the TensorFlow.js format. This allows you to build interactive web applications with client-side ML.
*   **ONNX (Open Neural Network Exchange):** ONNX is an open format for ML models. Converting your model to ONNX can be useful if you want to use it with a different deep learning framework (like PyTorch or Caffe) or a specific inference engine that supports ONNX.

---

### <mark>Things I Didn't Go Through (But You Should Explore)</mark>

*   **Post-Training Quantization:** This is a powerful technique to make your TFLite model even smaller and faster. Quantization reduces the precision of the model's weights (e.g., from 32-bit floating-point numbers to 8-bit integers). This can result in a 4x reduction in model size with a minimal drop in accuracy. The `TFLiteConverter` has options to enable different quantization strategies.

*   **Running Inference with the TFLite Interpreter:** A `.tflite` model is not used in the same way as a Keras model. You need to use the `tf.lite.Interpreter` to load the model and run predictions. Here is a quick peek at how it works:

    ```python
    import tensorflow as tf
    import numpy as np

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create some dummy input data
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    ```

## Small Project: Quantize and Compare

**Objective:** Convert a trained Keras model to TensorFlow Lite and see the impact of post-training quantization on the model's file size.

**Model:** You can use any `.keras` model you've trained in the previous days. The `intel_image_classifier.keras` from Day 28 or the `best_cat_dog_model.keras` from Day 27 are great choices.

**Steps:**

1.  **Load Your Keras Model:**
    *   Load your fully trained Keras model using `tf.keras.models.load_model()`.

2.  **Conversion #1: Standard TFLite Conversion:**
    *   Create a `TFLiteConverter` from your Keras model.
    *   Convert the model.
    *   Save the resulting `tflite_model` to a file named `model_float.tflite`.

3.  **Conversion #2: Conversion with Quantization:**
    *   Create another `TFLiteConverter` from the same Keras model.
    *   **Enable quantization:** Set the converter's optimizations flag.
    *   `converter.optimizations = [tf.lite.Optimize.DEFAULT]`
    *   Convert the model.
    *   Save the resulting quantized model to a file named `model_quant.tflite`.

4.  **Compare the File Sizes:**
    *   Use your operating system's file explorer or a command-line tool (`ls -lh`) to check the file sizes of `model_float.tflite` and `model_quant.tflite`.
    *   Calculate the difference. By what factor did quantization reduce the model size? (It should be close to 4x).

5.  **(Optional) Verify Accuracy:**
    *   To be thorough, you should also verify that quantization didn't significantly harm your model's accuracy.
    *   Write a helper function that takes a `.tflite` model file and a test dataset, runs inference on the entire dataset using the `TFLiteInterpreter` (as shown in the lesson), and calculates the accuracy.
    *   Run this function on both `model_float.tflite` and `model_quant.tflite`. How much did the accuracy change? For many models, the drop is minimal, making quantization a fantastic trade-off.

**Key Takeaway:** This project provides a tangible demonstration of a key model optimization technique. You will learn how to perform post-training quantization and see for yourself the significant reduction in model size it provides, a critical step for deploying models on resource-constrained devices.

This is a crucial step towards making your models truly useful. Happy converting!
