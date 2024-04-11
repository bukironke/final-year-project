from flask import Flask, request, render_template, session, redirect, url_for, make_response
import pickle
import numpy as np
import pandas as pd 
import shap
import pdfkit
import os
import fasttreeshap
import json

app = Flask(__name__)
app.secret_key = 'luxescore_secret'

#loads interest and credit scoring models
with open('models/defaulted_loan_model.pkl', 'rb') as f:
    defaulted_loan_model = pickle.load(f)

with open('models/interest_rate_model.pkl', 'rb') as f:
    interest_rate_model = pickle.load(f)

#function to define credit behaviour - max credit score = 900
def calculate_5_year_forecast(current_score, has_defaulted, employment_years, credit_history_years, owns_property, annual_income):
    scores = [current_score]
    uk_average_income = 33000
    
    scores.append(min(900, scores[-1] - 40 if has_defaulted else scores[-1] + 40))
    
    if credit_history_years < (employment_years / 2):
        scores.append(min(900, scores[-1] - 20))
    else:
        scores.append(min(900, scores[-1] + 20))
    
    scores.append(min(900, scores[-1] + 40 if owns_property else scores[-1]))
    
    scores.append(min(900, scores[-1] + 40 if annual_income > uk_average_income else scores[-1] - 20))
    
    return scores

#returns the landing page
@app.route('/')
def home():
    return render_template('index.html')

#returns the page where user enters  information
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        pass
    return render_template('loginpage.html')

#returns the user's calculated data
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received data:", data) 
    session['Name'] = data.get('Name', 'User')

    features_class_df = pd.DataFrame([[
        data['Age'],
        data['Income'],  
        data['HomeOwnership'], 
        data['EmploymentYears'],
        data['CreditHistoryYears']
    ]], columns=['Age', 'Income', 'HomeOwnership_encoded', 'Employment Years', 'Credit History Years'])

    features_regress_df = pd.DataFrame([[
        data['Age'],
        data['Income'],
        data['EmploymentYears'], 
        data['HomeOwnership'],   
        data['DefaultedLoan']    
    ]], columns=['Age', 'Income', 'Employment Years', 'HomeOwnership_encoded', 'Defaulted_encoded'])

    default_prediction = defaulted_loan_model.predict_proba(features_class_df)[0]
    print("Probability metrics:", default_prediction)
    threshold = 0.5
    probability_of_defaulting = default_prediction[1]
    credit_score = (1 - probability_of_defaulting) * 500 + 400
    credit_score = int(credit_score) 


    forecast_scores = calculate_5_year_forecast(
        credit_score,
        data.get('DefaultedLoan') == 1,  # Assuming '1' means they have defaulted
        data.get('EmploymentYears'),
        data.get('CreditHistoryYears'),
        data.get('HomeOwnership') !=  0,  # Rent is encoded as 0
        data.get('Income')
    )

    interest_rate_prediction = max(0, interest_rate_model.predict(features_regress_df)[0])
    interest_rate_prediction = round(interest_rate_prediction, 2)

    #renders data into sessions
    session['default'] = 'will default' if probability_of_defaulting > threshold else 'will not default'
    session['credit_score'] = credit_score
    session['interest_rate'] = interest_rate_prediction
    session['forecast_scores'] = json.dumps(forecast_scores)

    # Print the details
    print(f"Name: {session['Name']}")
    print(f"Default Status: {session['default']}")
    print(f"Credit Score: {session['credit_score']}")
    print(f"Interest Rate: {session['interest_rate']}%")
    print(f"Credit Score changes in 5 years: {session['forecast_scores']}")

    #calculating shap values 
    explainer = fasttreeshap.Explainer(defaulted_loan_model)
    shap_values = explainer.shap_values(features_class_df)
    print("SHAP values:", shap_values)
    predicted_class = defaulted_loan_model.predict(features_class_df)[0]
    shap_values_for_user = shap_values[predicted_class][0]
    print("SHAP values before normalisation:", shap_values_for_user)
    abs_shap_values = np.abs(shap_values_for_user)
    normalised_shap_values = 100 * abs_shap_values / np.sum(abs_shap_values)
    normalised_shap_values_rounded = np.round(normalised_shap_values, 2)
    print("Normalised SHAP values:", normalised_shap_values)
    print("Rounded SHAP values:", normalised_shap_values_rounded)
    feature_names = ['Age', 'Income', 'HomeOwnership', 'EmploymentYears', 'CreditHistoryYears'] 
    shap_values_for_chart = list(zip(feature_names, normalised_shap_values_rounded.tolist()))
    shap_values_for_pdf = list(zip(feature_names, normalised_shap_values_rounded.tolist()))
    print("Feature with its importance:", shap_values_for_chart)
    session['shap_values_for_chart'] = json.dumps(shap_values_for_chart)
    session['shap_values_for_pdf'] = shap_values_for_pdf
    
    return redirect(url_for('show_results'))

#displays page with results
@app.route('/results')
def show_results():
    default = session.get('default', 'Error')
    credit_score = session.get('credit_score', 0)
    interest_rate = session.get('interest_rate', 0)
    name = session.get('Name', 'User')

    shap_values_for_chart = session.get('shap_values_for_chart', [])
    forecast_scores = session.get('forecast_scores', []) 
    print("Sent!") 
    return render_template('creditscoreResult.html', shap_values_for_chart=shap_values_for_chart, forecast_scores=forecast_scores, default=default, credit_score=credit_score, interest_rate=interest_rate, name=name)

#returns pdf with user's information
@app.route('/generate_pdf', methods=['GET'])
def generate_pdf():
    default = session.get('default', 'Error')
    credit_score = session.get('credit_score', 0)
    interest_rate = session.get('interest_rate', 0)
    name = session.get('Name', 'User')
    shap_values_for_pdf = session.get('shap_values_for_pdf', [])

    html_content = render_template('pdf_template.html', shap_values_for_pdf=shap_values_for_pdf, default=default, credit_score=credit_score, interest_rate=interest_rate, name=name)
    pdf_filename = f"{name}'s_luxescore_report.pdf"

    pdf_content = pdfkit.from_string(html_content, False)

    response = make_response(pdf_content)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={pdf_filename}' 
    print("PDF Generated!") 
    return response


if __name__ == '__main__':
    app.run(debug=True)