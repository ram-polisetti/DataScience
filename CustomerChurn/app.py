from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained machine learning model
model = joblib.load('path_to_your_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract parameters from the request
    data = request.get_json()

    # Ensure all parameters are present
    required_parameters = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                           'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'Churn', 
                           'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes', 
                           'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 
                           'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
                           'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
                           'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
                           'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes', 
                           'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes', 
                           'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
                           'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year', 
                           'PaymentMethod_Bank transfer', 'PaymentMethod_Credit card', 
                           'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

    for param in required_parameters:
        if param not in data:
            return jsonify({'error': f'Missing parameter: {param}'}), 400

    # Convert parameters to the format expected by the model
    input_data = np.array([data[param] for param in required_parameters]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=9696)
