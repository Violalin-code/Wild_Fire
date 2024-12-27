# Wildfire Prediction App 🌳🔥
## Overview
This project implements a Wildfire Prediction System that predicts the likelihood of a wildfire occurring based on environmental factors, and estimates the fire size, duration, and suppression costs. The system uses pre-trained machine learning models for classification and regression tasks and is powered by Gradio for a user-friendly web interface.

## Table of Contents
- [ML Models](#ML-Models)
- [Features](#Features)
- [How to Run the Project](#How-to-Run)
- [Project Workflow](#Project-Workflow)

## ML Models
Classification Model: Logistic Regression, chosen for its low RMSE (0.5514), predicts whether a wildfire will occur based on environmental variables.
Regression Model: Random Forest Regressor, used for predicting the size, duration, and suppression cost of the wildfire.

## Features
- Wildfire Occurrence Prediction: Predicts whether a wildfire will occur based on environmental inputs such as temperature, humidity, and geographical factors.
- Fire Characteristics Estimation: Estimates the size (in hectares), duration (in hours), and suppression costs of the predicted wildfire.
- Pre-trained Models: Uses pre-trained models saved in pickle files for prediction.
- Interactive Interface: Gradio provides an easy-to-use web interface, allowing users to input data and receive real-time predictions.

## How to Run
- Install Dependencies: Ensure that the required libraries are installed. The code will automatically install scikit-learn if it's missing.
- Prepare Files: Place the following files in the same directory as the script:
- best_classification_model.pkl: Pre-trained classification model for predicting wildfire occurrence.
- best_regression_model.pkl: Pre-trained regression model for predicting fire size, duration, and suppression cost.
- label_encoder.pkl: (Optional) Label encoder for decoding categorical predictions.
- Run the Script: Execute the script to start the app.

GUI:https://huggingface.co/spaces/vjl004/Wild_Fire

## Project Files
- dataset.csv: The input dataset containing historical data on fire incidents and environmental factors.
- label_encoder.pkl: A saved label encoder to handle categorical labels (if used in the dataset).
- best_classification_model.pkl: The best-performing classification model for predicting wildfire occurrence.
- best_regression_model.pkl: The best-performing regression model for predicting fire size, duration, and suppression costs.

## Project Workflow
### 1. Data Preprocessing
- Features: Environmental and geographical variables like temperature, humidity, wind speed, and location.
- Classification: Fire occurrence (binary: will the wildfire happen or not).
- Regression: Fire size (hectares), fire duration (hours), suppression cost (USD).
- Handling Missing Data: Rows with missing target values (e.g., fire occurrence) are removed, while regression targets are handled using imputation if necessary.

### 2. Encoding and Scaling
- One-Hot Encoding: Categorical features (if any) are one-hot encoded to convert them into numeric values.
- Standardization: Features are standardized for models like Support Vector Machine (SVM), which require scaled data for optimal performance.
- Classification Models: Several classification models are trained to predict wildfire occurrence, including:

### 3. Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- Support Vector Machine (SVM)
- Each model is evaluated using Root Mean Squared Error (RMSE) to assess the accuracy of predictions. The model with the lowest RMSE on the test data is selected as the best model.

### 4. Regression Models
- For predicting fire size, duration, and suppression cost, a Random Forest Regressor is used. This model is trained to predict multiple outputs simultaneously. The regression model’s performance is evaluated using RMSE, and the best-performing model is selected for deployment.

### 5. Best Model Selection and Saving
- Best Classification Model: The best-performing classification model is saved as best_classification_model.pkl for future predictions.
- Best Regression Model: The best-performing regression model is saved as best_regression_model.pkl.
