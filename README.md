# Customer Churn Prediction for Telecom Industry

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Choose an appropriate license -->
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](.) <!-- Placeholder -->

This project focuses on predicting customer churn within the telecom industry using machine learning. By analyzing customer demographics, usage patterns, and account information, we aim to identify customers at risk of leaving and understand the key drivers behind churn, ultimately enabling proactive retention strategies.

---

## Table of Contents

*   [1. Introduction](#1-introduction)
    *   [Project Objective](#project-objective)
    *   [Key Questions](#key-questions)
*   [2. Business Understanding](#2-business-understanding)
    *   [Importance of Churn Prediction](#importance-of-churn-prediction)
    *   [Challenges Addressed by ML](#challenges-addressed-by-ml)
*   [3. Data Understanding](#3-data-understanding)
    *   [Dataset Overview](#dataset-overview)
    *   [Key Features](#key-features)
*   [4. Methodology (CRISP-DM)](#4-methodology-crisp-dm)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Model Building & Tuning](#model-building--tuning)
    *   [Model Evaluation](#model-evaluation)
    *   [Model Selection](#model-selection)
*   [5. Model Results](#5-model-results)
*   [6. Deployment](#6-deployment)
*   [7. Project Structure](#7-project-structure)
*   [8. Technology Stack](#8-technology-stack)
*   [9. Getting Started](#9-getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
*   [10. Usage](#10-usage)
*   [11. License](#11-license)
*   [12. Acknowledgments](#12-acknowledgments)

---

## 1. Introduction

### Project Objective

In the highly competitive telecom industry, customer retention is paramount. This project leverages machine learning to predict customer churn, enabling telecom companies to:

1.  Identify patterns and factors contributing to customer churn.
2.  Build accurate models to predict which customers are likely to churn.
3.  Provide actionable insights to implement effective retention strategies and reduce revenue loss.

### Key Questions

*   What are the primary factors driving customer churn in this dataset?
*   How accurately can machine learning models predict customer churn?
*   How can the model's predictions be interpreted to guide business decisions for retention?

---

## 2. Business Understanding

### Importance of Churn Prediction

Retaining existing customers is significantly more cost-effective (up to 5x cheaper) than acquiring new ones in the telecom sector. Predicting and mitigating churn leads to:

*   Increased Customer Lifetime Value (CLTV).
*   Improved profitability via targeted retention efforts.
*   Enhanced brand loyalty and a stronger competitive position.

### Challenges Addressed by ML

Traditional churn analysis methods can be slow, limited in scope, and reactive. Machine learning offers advantages by:

*   Automating pattern discovery in large datasets.
*   Handling complex, non-linear customer behaviors.
*   Providing scalable and potentially real-time predictions for proactive interventions.

---

## 3. Data Understanding

### Dataset Overview

The project utilizes two datasets derived from a common telecom churn source:

*   `churn-bigml-80.csv`: **Training set** (80% of data) used to train the machine learning models.
*   `churn-bigml-20.csv`: **Testing set** (20% of data) used to evaluate the performance of the trained models on unseen data.

*(Source: Often attributed to the BigML churn dataset, commonly used for educational/benchmark purposes).*

### Key Features

The dataset includes features crucial for churn analysis:

*   **Customer Demographics:** `State`, `Area code`
*   **Account Information:** `Account length`
*   **Service Usage:**
    *   Plans: `International plan`, `Voice mail plan`
    *   Call Metrics: `Total day minutes/calls/charge`, `Total eve minutes/calls/charge`, `Total night minutes/calls/charge`, `Total intl minutes/calls/charge`
    *   Service Interaction: `Customer service calls`
*   **Target Variable:** `Churn` (Yes/No or True/False)

---

## 4. Methodology (CRISP-DM)

The project followed the Cross-Industry Standard Process for Data Mining (CRISP-DM) framework:

1.  **Business Understanding:** (Covered above)
2.  **Data Understanding:** Initial exploration of data, feature types, distributions, and target variable imbalance. (Covered above)
3.  **Data Preprocessing:**
    *   Handling missing values (if any).
    *   Encoding categorical features (e.g., State, Plans) using techniques like One-Hot Encoding or Label Encoding.
    *   Feature Scaling (e.g., StandardScaler, MinMaxScaler) for models sensitive to feature magnitudes (like KNN, SVM, Neural Networks).
    *   Handling class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique) where tested.
4.  **Model Building & Tuning:** A variety of classification models were trained and evaluated:
    *   Logistic Regression (with and without SMOTE)
    *   K-Nearest Neighbors (KNN) (with and without SMOTE)
    *   Support Vector Machine (SVM) with different kernels (Linear, Polynomial, RBF, Sigmoid)
    *   Decision Tree
    *   Random Forest
    *   AdaBoost
    *   XGBoost
    *   Neural Networks
    *   **Hyperparameter Tuning:** `GridSearchCV` was employed to find the optimal hyperparameters for each model based on cross-validation performance.
5.  **Model Evaluation:** Models were compared using:
    *   **Confusion Matrix:** To analyze True Positives, False Positives, True Negatives, and False Negatives. Metrics like Precision, Recall, F1-Score, and Accuracy can be derived.
    *   **ROC AUC Curve:** To evaluate the model's ability to distinguish between the churn and non-churn classes across different thresholds.
6.  **Model Selection:** Based on comparative evaluation on both training and testing sets (primarily focusing on test set performance using ROC AUC and other relevant metrics like Recall for churn class), the best-performing model was chosen.

---

## 5. Model Results

After training, tuning, and evaluating multiple models, **XGBoost** demonstrated the best performance on the test set according to the chosen evaluation metrics (specifically ROC AUC and likely a balance of precision/recall). It provided the most reliable predictions for identifying customers likely to churn on unseen data.

---

## 6. Deployment

The selected XGBoost model was exported (likely as a `.pkl` or `.joblib` file) and deployed as a simple web application.

*   **Interface:** The application provides a user-friendly form where inputs corresponding to the model's features can be entered (e.g., State, Account Length, Plan Status, Call Minutes/Counts, Customer Service Calls).
*   **Prediction:** Upon submitting the form, the application uses the loaded XGBoost model to predict whether the customer described by the input features is likely to churn or not.
*   **Output:** A clear message indicates the prediction ("The customer is likely to churn." or "The customer will not churn.").

**(Screenshots)**
*(Consider checking the test_images folder)*

### Implementation
*   Download the project and unzip it
*   Access the deployment/ folder
*   Activate the virtual environment
```bash
...deployment> venv\Scripts\activate
```
*   Run the application
```bash
...deployment> python app.py to
```
