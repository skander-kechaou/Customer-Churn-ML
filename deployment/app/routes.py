from flask import request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

model = joblib.load('modelXGB.pkl')  # Load the trained XGBoost model
scaler = joblib.load('scaler.pkl')  # Load the scaler used for feature scaling
enc = joblib.load('frequencyEnc.pkl')

def configure_routes(app):
    @app.route('/')
    def default():
        return render_template('xgboost.html')

    @app.route('/xgb')
    def xgb_open():
        return render_template('xgboost.html')
    
    @app.route('/predict', methods=['POST'])
    def predict():
        # Get the form input values for all variables
        international_plan = int(request.form['international_plan'])
        voice_mail_plan = int(request.form['voice_mail_plan'])  # Assuming this is a 0/1 input
        avg_intl_duration = float(request.form['avg_intl_duration'])
        total_day_minutes = float(request.form['total_day_minutes'])
        total_eve_minutes = float(request.form['total_eve_minutes'])
        total_night_minutes = float(request.form['total_night_minutes'])
        total_intl_minutes = float(request.form['total_intl_minutes'])
        total_intl_calls = int(request.form['total_intl_calls'])
        customer_service_calls = int(request.form['customer_service_calls'])
        account_length = float(request.form['account_length'])
        total_day_calls = int(request.form['total_day_calls'])
        total_night_calls = int(request.form['total_night_calls'])
        total_eve_calls = int(request.form['total_eve_calls'])

        # Create the input data dictionary for all variables
        data = {
            'International plan': [international_plan],
            'Voice mail plan': [voice_mail_plan],
            'Avg intl duration': [avg_intl_duration],
            'Account length': [account_length],
            'Total day minutes': [total_day_minutes],
            'Total day calls': [total_day_calls],
            'Total eve minutes': [total_eve_minutes],
            'Total eve calls': [total_eve_calls],
            'Total night minutes': [total_night_minutes],
            'Total night calls': [total_night_calls],
            'Total intl minutes': [total_intl_minutes],
            'Total intl calls': [total_intl_calls],
            'Customer service calls': [customer_service_calls],
        }

        # Create a DataFrame from the input data
        input_df = pd.DataFrame(data)

        # Apply scaling to the input data using the scaler (if the model was trained with scaling)
        input_df = scaler.transform(input_df)

        # Predict the outcome using the XGBoost model
        prediction = model.predict(input_df)
        
        # Prepare the result message based on the prediction
        if prediction == 0:
            result = "The customer will not churn."
        else:
            result = "The customer is likely to churn."        
        return jsonify({'result': result})
