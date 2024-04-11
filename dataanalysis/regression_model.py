#importing relevant libraries
#Please note that most of the EDA and trial and error is in the notebook scripts (mlmodel1, 2 and 3), this is just to give a more simplistic illustration of my model
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("dataanalysis/creditriskdataset.csv")

#DATA CLEANING

#function for dropping columns/rows
def clean_dataset(data):
    #dropping rows with missing values
    data.dropna(subset=['person_emp_length', 'loan_int_rate'], inplace=True)
    #drop columns
    data.drop(columns=['loan_amnt', 'loan_status', 'loan_percent_income', 'loan_grade'], inplace=True)
    #average life expectancy in uk is 81 in the UK
    data = data[data['person_age']<81]
    #average retirement age is 66 currently, filtering employment length
    emp_max = 66 - 18
    data = data[data['person_emp_length'] <= emp_max]
    #since the application gives out personal loans, only want to keep instances of loans which are classified as 'personal'
    data = data[~data['loan_intent'].isin(['EDUCATION', 'MEDICAL', 'VENTURE', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])]
    data.drop(columns=['loan_intent'], inplace=True)
    #'other' is hard to quantify when gathering user information
    data = data[data['person_home_ownership'] != 'OTHER']
    print('Dataset has been cleaned. Current shape:', data.shape)
    print('Amount of null values in each column is:')
    print(data.isnull().sum())      
    return data


#DATA TRANSFORMATION

def transform_dataset(data):
    #renaming columns
    data = data.rename(columns= {"person_age": "Age", "person_income": "Income", "person_emp_length": "Employment Years", "cb_person_cred_hist_length": "Credit History Years",
    "person_home_ownership": "Home Ownership", "cb_person_default_on_file": "Defaulted on Loan", "loan_int_rate": "Interest Rate"})
    #encoding categorical columns for model and EDA
    label_encoder = LabelEncoder()
    data['HomeOwnership_encoded'] = label_encoder.fit_transform(data['Home Ownership'])
    data['Defaulted_encoded'] = label_encoder.fit_transform(data['Defaulted on Loan'])
    print('Data transformation has been completed.')
    return data  

#MODEL BUILDING and EVALUATION

#fitting feature and target variable for model
numerical_columns = ['Age', 'Income', 'Employment Years', 'HomeOwnership_encoded', 'Defaulted_encoded']
X = data[numerical_columns]
y = data['Interest Rate']

#creating training and testing split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#building model and then printing out its OOB, MSE and R SQUARED metric
best_regressor = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=2, min_samples_split=10, max_features='log2', random_state=0, oob_score=True)

best_regressor.fit(X_train, y_train)

oob_score = best_regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')

predictions_train = best_regressor.predict(X_train)

mse_train = mean_squared_error(y_train, predictions_train)
print(f'Mean Squared Error (Training): {mse_train}')

r2_train = r2_score(y_train, predictions_train)
print(f'R-squared (Training): {r2_train}')

predictions_test = best_regressor.predict(X_test)

mse_test = mean_squared_error(y_test, predictions_test)
print(f'Mean Squared Error (Test): {mse_test}')

r2_test = r2_score(y_test, predictions_test)
print(f'R-squared (Test): {r2_test}')