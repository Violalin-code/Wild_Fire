README: Wildfire Prediction App

OVERVIEW

This project implements a Wildfire Prediction System that predicts the likelihood of a wildfire occurring based on environmental factors, as well as estimates fire size, duration, and suppression cost. 
The app uses multiple pre-trained models to provide these predictions and is powered by Gradio for a user-friendly web interface.

ML Model: Logistic Regression with RMSE of 0.5514 is the best classification model. Typically Logistic Regression is specifically designed for binary classification.

FEATURES

Predicts whether a wildfire will occur based on environmental inputs.
Provides estimates for the size, duration, and suppression cost of the predicted wildfire.
Uses pre-trained models stored in pickle files for prediction.
Interactive Gradio interface allowing users to input data and receive predictions.

HOW TO RUN:

1. Ensure all required dependencies are installed. The script will install scikit-learn if it is not found on the system.

2. Place the pre-trained models and label encoder files in the same directory as the script.

3. Run the script

GUI:https://huggingface.co/spaces/vjl004/Wild_Fire
