import subprocess
import sys

# Force install scikit-learn if not found
try:
    import sklearn
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import sklearn  # Import again after installation

import gradio as gr
import pandas as pd
import pickle

# Load the pre-trained model for Fire Occurrence
with open('best_model_occurrence.pkl', 'rb') as model_file:
    model_occurrence = pickle.load(model_file)

# Load the pre-trained models for regression targets (Fire Size, Fire Duration, Suppression Cost)
with open('best_model_fire size (hectares).pkl', 'rb') as model_file_size:
    model_size = pickle.load(model_file_size)

with open('best_model_fire duration (hours).pkl', 'rb') as model_file_duration:
    model_duration = pickle.load(model_file_duration)

with open('best_model_suppression cost ($).pkl', 'rb') as model_file_cost:
    model_cost = pickle.load(model_file_cost)

# Load the label encoder
with open('label_encoder (5).pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

def predict_wildfire(Temperature, Humidity, Wind_Speed, Rainfall, Fuel_Moisture, Vegetation, Slope, Region):
    # Creating input DataFrame for the model
    input_data = pd.DataFrame({
        'Temperature (°C)': [Temperature],
        'Humidity (%)': [Humidity],
        'Wind Speed (km/h)': [Wind_Speed],
        'Rainfall (mm)': [Rainfall],
        'Fuel Moisture (%)': [Fuel_Moisture],
        'Vegetation Type': [Vegetation],
        'Slope (%)': [Slope],
        'Region': [Region]
    })

    # One-hot encode the input data (ensure it matches the training data)
    input_encoded = pd.get_dummies(input_data)

    # Align columns with the training data (required columns)
    required_columns = model_occurrence.feature_names_in_  # Get the feature columns from the model
    for col in required_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[required_columns]

    # Make the prediction for Fire Occurrence (Yes/No)
    fire_occurrence_prediction = model_occurrence.predict(input_encoded)[0]

    # Map 1 to 'Yes' and 0 to 'No'
    fire_occurrence = 'Yes' if fire_occurrence_prediction == 1 else 'No'

    # Predict for other targets (Fire Size, Fire Duration, Suppression Cost)
    fire_size_prediction = model_size.predict(input_encoded)[0]  # Regression for Fire Size
    fire_duration_prediction = model_duration.predict(input_encoded)[0]  # Regression for Fire Duration
    suppression_cost_prediction = model_cost.predict(input_encoded)[0]  # Regression for Suppression Cost

    return (fire_occurrence, fire_size_prediction, fire_duration_prediction, suppression_cost_prediction)

# Gradio Interface using components
interface = gr.Interface(
    fn=predict_wildfire,
    inputs=[
        gr.Textbox(label="Temperature (°C)"),
        gr.Textbox(label="Humidity (%)"),
        gr.Textbox(label="Wind Speed (km/h)"),
        gr.Textbox(label="Rainfall (mm)"),
        gr.Textbox(label="Fuel Moisture (%)"),
        gr.Dropdown(['Grassland', 'Forest', 'Shrubland'], label="Vegetation Type"),
        gr.Textbox(label="Slope (%)"),
        gr.Dropdown(['North', 'South', 'East', 'West'], label="Region")
    ],
    outputs=[
        gr.Textbox(label="Fire Occurrence ('Yes' or 'No')"),
        gr.Textbox(label="Fire Size (hectares)"),
        gr.Textbox(label="Fire Duration (hours)"),
        gr.Textbox(label="Suppression Cost ($)")
    ],
    title="Wildfire Prediction"
)

if __name__ == "__main__":
    interface.launch()
