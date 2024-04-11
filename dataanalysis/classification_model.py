#Please note that most of the EDA and trial and error is in the notebook scripts (mlmodel1, 2 and 3), this is just to give a more simplistic illustration of my model
#importing relevant libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import fasttreeshap
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle


data = pd.read_csv("dataanalysis/creditriskdataset.csv")

#DATA CLEANING

#function for dropping columns/rows
def clean_dataset(data):
    #dropping rows with missing values
    data.dropna(subset=['person_emp_length', 'loan_int_rate'], inplace=True)
    #drop columns
    data.drop(columns=['loan_amnt', 'loan_status', 'loan_percent_income', 'loan_grade'], inplace=True)
    #average life expectancy in uk is 81 in the UK
    data = data[data['person_age'] < 81]
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
    "person_home_ownership": "Home Ownership", "cb_person_default_on_file": "Defaulted on Loan"})
    #encoding categorical columns for model and EDA
    label_encoder = LabelEncoder()
    data['HomeOwnership_encoded'] = label_encoder.fit_transform(data['Home Ownership'])
    data['Defaulted_encoded'] = label_encoder.fit_transform(data['Defaulted on Loan'])
    print('Data transformation has been completed.')
    return data  
    
#MODEL BUILDING, EVALUATION AND SHAPLEY VALUES

#fitting feature and target variable for model
feature_columns = ['Age', 'Income', 'HomeOwnership_encoded', 'Employment Years', 'Credit History Years']
X = data[feature_columns]
y = data['Defaulted_encoded']

#imbalance of classes - so random oversampling to make both classes equal
ros = RandomOverSampler(sampling_strategy=1)
X_res, y_res = ros.fit_resample(X, y)
y_res.value_counts()

#creating training and testing split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

#building model and then printing out its classification report, accuracy and confusion matrix
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(X_train, y_train)

y_pred_rf = rf_classifier.predict(X_test)

print(classification_report(y_test, y_pred_rf))

accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:")
print(conf_matrix)

#cross validating the model to see its performance against untrained data
cv = StratifiedKFold(n_splits=5)
scores = cross_val_score(rf_classifier, X, y, cv=cv, scoring='accuracy')

print("Cross-Validation Scores:", scores)
print("Average score:", np.mean(scores))

#save the model in pickle file so can be deployed onto app
model_loan_file = "defaulted_loan_model.pkl"
with open(model_loan_file, 'wb') as file:
    pickle.dump(rf_classifier, file)

#calculating shap values - utilising the original module and the 'fasttreeshap' module - visualisation of the shap values for both classes is in mlmodel2.ipynb

#original module 
explainer = shap.TreeExplainer(rf_classifier)
shap_values = explainer.shap_values(X)

#shap values saved as a list so it is 3d rather than 2d automatically so need to convert it into an array before 
shap_values_array = np.array(shap_values)

#create an explanation obj - for the not defaulted loan
exp = shap.Explanation(values=shap_values_array[:, 0, :], base_values=explainer.expected_value[0], data=X)
#create an explanation obj - for the defaulted loan
exp = shap.Explanation(values=shap_values_array[:, 1, :], base_values=explainer.expected_value[1], data=X)

#fasttreeshap module 
#3 versions of fasttreeshap which reduces script run time - from 2m down to 11-12s 
#version 1
explainer = fasttreeshap.TreeExplainer(rf_classifier, algorithm = "v1")
#version 2
explainer = fasttreeshap.TreeExplainer(rf_classifier, algorithm = "v2")
#auto - fastest at runtime of 11 secs
explainer = fasttreeshap.TreeExplainer(rf_classifier, algorithm = "auto")
shap_values = explainer.shap_values(X)

#function to get the numerical contributions of each feature to the model decision based on the shapley value calculation
feature_cols = ['Age', 'Income', 'HomeOwnership_encoded', 'Employment Years', 'Credit History Years']

def shap_percentages(shap_values_array):
    shap_default = shap_values_array[1, :, :] #for not defualting, the 1 would be 0

    av_shapvalues = np.mean(np.abs(shap_default), axis=0)
    total_shapvalues = np.sum(av_shapvalues)

    print ("Feature contributions for defaulting on a loan:")
    for i, feature in enumerate(feature_cols):
      contribution_percent = (av_shapvalues[i] / total_shapvalues) * 100
      print (f"{feature} contributed to {contribution_percent:.2f}% of the decision")

shap_values_array_example = np.random.rand(2, 1, len(feature_cols)) 
shap_percentages(shap_values_array_example)
