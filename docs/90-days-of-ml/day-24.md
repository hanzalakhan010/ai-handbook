---
id: day-24
title: 'Day 24: Post-Deployment and Advanced Evaluation Metrics'
---

## Day 24: Post-Deployment and Advanced Evaluation Metrics

Today, we'll focus on crucial aspects of machine learning models that come into play after training: **post-deployment considerations** and a deeper dive into **evaluation metrics**, especially for critical applications like medical diagnosis.

### Key Learnings

1.  **NumPy Array Consistency:**
    *   **Rule:** NumPy arrays, especially when used in deep learning, should always have consistent data types and shapes across all dimensions to prevent errors and ensure efficient computation.

### Post-Deployment Considerations: Monitoring and Maintenance

Deploying a machine learning model is not the end of the journey; it's just the beginning. Continuous monitoring and planned updates are vital for maintaining model performance and reliability in real-world scenarios.

*   **Monitoring:**
    *   **Track Inference Latency:** How long does it take for the model to make a prediction? High latency can degrade user experience.
    *   **Memory Usage:** How much memory does the model consume on the deployment device? Crucial for edge devices.
    *   **Battery Impact:** For mobile applications, model inference shouldn't drain the device's battery excessively.
    *   **Tools:** Utilize tools like TensorFlow Model Analysis (TFMA) to track edge metrics and ensure your model behaves as expected in production.

*   **Updates:**
    *   **Over-the-Air (OTA) Updates:** Deploy new model versions seamlessly via OTA updates, minimizing disruption to users.

*   **Privacy:**
    *   **On-Device Processing:** For sensitive data, ensure all data processing and inference happen on the device itself, avoiding server calls to maintain user privacy.

### Why TensorFlow Lite?

TensorFlow Lite is designed for on-device machine learning and addresses many of the post-deployment challenges:

*   **Lightweight:** Optimized for low latency and small binary size, making it suitable for resource-constrained devices.
*   **Cross-Platform:** Supports various platforms including Android, iOS, Linux-based devices (like Raspberry Pi), and microcontrollers.
*   **Hardware Acceleration:** Leverages specialized hardware (GPUs, NPUs, Android Neural Networks API) for faster inference.
*   **Privacy:** Enables local inference without sending data to servers, enhancing user privacy.

### Understanding Precision and Recall in Detail

The "good" range for **recall** and **precision** depends entirely on your **use case** and the **cost of errors**. There’s no universal "ideal" range, but here’s a framework to evaluate them:

---

#### **1. High-Risk Scenarios (e.g., Healthcare, Fraud Detection)**

*   **Recall (Sensitivity):**
    *   **Aim for ≥90%** if missing a positive case is critical (e.g., cancer detection, pneumonia diagnosis).
    *   *Example:* A pneumonia detector with **95% recall** misses only 5% of true cases (critical for saving lives).

*   **Precision:**
    *   **Accept lower precision (e.g., 70-80%)** if false positives are less harmful than false negatives.
    *   *Example:* Flagging 20% healthy patients as "potential pneumonia" for further tests is better than missing 5% true cases.

---

#### **2. Low-Stakes Scenarios (e.g., Spam Detection)**

*   **Precision:**
    *   **Aim for ≥90%** if false positives are annoying but not harmful.
    *   *Example:* Classifying legitimate emails as spam (false positives) frustrates users.

*   **Recall:**
    *   **Accept lower recall (e.g., 80%)** if missing some positives is tolerable.
    *   *Example:* Missing 20% spam emails is acceptable if inboxes stay mostly clean.

---

#### **3. Balanced Scenarios (e.g., Customer Churn)**

*   **Trade-off:**
    *   Use the **F1-score** (harmonic mean of precision and recall) to balance both.
    *   *Example:* An F1-score of **≥0.85** is strong for most business applications.

---

#### **4. Context-Specific Guidelines**

| **Use Case**               | Priority          | Target Recall | Target Precision |
| :------------------------- | :---------------- | :------------ | :--------------- |
| **Medical Diagnosis**      | Recall            | ≥90%          | ≥70%             |
| **Fraud Detection**        | Precision         | ≥80%          | ≥90%             |
| **Legal Document Review**  | Both              | ≥85%          | ≥85%             |
| **Marketing Lead Scoring** | Precision         | ≥75%          | ≥90%             |

---

#### **Key Considerations**

1.  **Cost of False Negatives vs. False Positives:**
    *   **High cost of FN** (e.g., medical missed diagnoses): Prioritize recall.
    *   **High cost of FP** (e.g., spam filters): Prioritize precision.

2.  **Class Imbalance:**
    *   If the positive class is rare (e.g., fraud), even **50% recall** might be impressive if the baseline (random guessing) is 1%.

3.  **Business/Stakeholder Needs:**
    *   A model with **80% recall/70% precision** might be "good enough" if manual review can handle false positives.

---

    _ (Content removed due to repeated MDX parsing issues. This section originally contained information on "Tools to Evaluate Trade-offs" including Precision-Recall Curves and F-beta Score, along with a detailed "Example for Pneumonia Detection" and "Final Takeaway".)_

## Small Project: Optimizing for Recall in Pneumonia Detection

**Objective:** Apply your knowledge of the precision-recall trade-off to tune a model's decision threshold for a high-stakes medical application.

**Scenario:** For pneumonia detection, a **False Negative** (missing a case of pneumonia) is far more dangerous than a **False Positive** (flagging a healthy patient for a second look). Therefore, our primary goal is to maximize **recall**.

**Dataset:** We'll use the [Pneumonia X-Ray Dataset from Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). For simplicity, you can use a pre-trained model you've already built or download one. The key part of this exercise is the evaluation, not the training.

**Steps:**

1.  **Load Your Model and Test Data:**
    *   Load a pre-trained CNN model capable of classifying the pneumonia images.
    *   Load your test dataset (images and true labels). Make sure it is preprocessed in the same way the model expects.

2.  **Get Prediction Probabilities:**
    *   Instead of getting final class predictions (`model.predict`), get the raw prediction probabilities for the "PNEUMONIA" class using `model.predict()`. This will give you an array of values between 0 and 1.

3.  **Evaluate at Different Thresholds:**
    *   The default threshold is 0.5. Let's see how changing it affects our metrics.
    *   Create a loop that iterates through different thresholds, from 0.1 to 0.9.
    *   `thresholds = np.arange(0.1, 1.0, 0.05)`
    *   Inside the loop, for each `threshold`:
        *   Convert your probabilities into class predictions: `predicted_labels = (probabilities > threshold).astype(int)`
        *   Calculate the `confusion_matrix`.
        *   Calculate `precision_score` and `recall_score`.
        *   Store the precision and recall for each threshold.

4.  **Find the Optimal Threshold for High Recall:**
    *   Look at your stored results. What is the lowest threshold you can use that achieves a **recall of at least 95%**?
    *   At that threshold, what is the corresponding precision? You will likely see that to get very high recall, you have to accept lower precision.

5.  **Visualize the Trade-off:**
    *   Plot the Precision-Recall Curve using `PrecisionRecallDisplay.from_predictions()`. This plot perfectly visualizes the trade-off you just calculated. You can see how precision drops as recall increases.

**Key Takeaway:** This project makes the abstract concept of the precision-recall trade-off concrete. You will learn that a deployed model is not just the trained weights; the **decision threshold** is a critical, tunable hyperparameter that must be set based on the specific business or clinical needs of the application.
