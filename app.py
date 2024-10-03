from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import load_model
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load the hybrid model

hybrid_model = load_model('hybrid_model.h5')


# Define the list of numerical features
numerical_features = ['Age', 'Gestation in previous Pregnancy', 'BMI', 'HDL', 'Sys BP', 'OGTT', 'Hemoglobin']

@app.route('/')
def home():
    return render_template("index.html", result=None)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        age = int(request.form.get("age"))
        gestation = int(request.form.get("gestation"))
        bmi = float(request.form.get("bmi"))
        hdl = float(request.form.get("hdl"))
        pcos = int(request.form.get("pcos"))
        sys_bp = float(request.form.get("sys_bp"))
        dia_bp = int(request.form.get("dia_bp"))
        ogtt = float(request.form.get("ogtt"))
        hemoglobin = float(request.form.get("hemoglobin"))
        prediabetes = int(request.form.get("prediabetes"))

        # Create a DataFrame from the form data
        input_data = pd.DataFrame({
            'Name': [name],
            'Age': [age],
            'Gestation in previous Pregnancy': [gestation],
            'BMI': [bmi],
            'HDL': [hdl],
            'pcos': [pcos],
            'Sys BP': [sys_bp],
            'Dia BP': [dia_bp],
            'OGTT': [ogtt],
            'Hemoglobin': [hemoglobin],
            'Prediabetes': [prediabetes],
        })

        # Validate and transform the input data
        try:

            # Select relevant features
            selected_features = ['Age', 'Gestation in previous Pregnancy', 'BMI', 'HDL', 'Sys BP', 'OGTT', 'Hemoglobin',
                                 'pcos', 'Dia BP', 'Prediabetes']
            input_data = input_data[selected_features]

            # Make predictions using the model
            prediction = hybrid_model.predict([input_data.values, input_data.values])

            print("Prediction:", prediction)

            prediction_value = prediction[0][0]
            print("Prediction Value:", prediction_value)

            # Convert prediction to human-readable format
            result = 'GDM' if (prediction_value >= 0.99997) else 'Non GDM'

            print("Result:", result)

            return render_template("index.html", result=result, **request.form)

        except Exception as e:
            print(f"An error occurred: {e}")
            return render_template("index.html", error="An error occurred while processing the request.")



if __name__ == "__main__":
    app.run(debug=True)
