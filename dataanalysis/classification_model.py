#Please note that most of the EDA and trial and error is in the notebook scripts (mlmodel1, 2 and 3), this is just to give a more simplistic illustration of my model
#importing relevant libraries
import pandas as pd 
import numpy as np
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

def load_data(file_path):
    return pd.read_csv(file_path)

#function to perform data cleaning progress
def clean_data(data):
    #dropping rows with missing values
    data.dropna(subset=['person_emp_length', 'loan_int_rate'], inplace=True)
    #drop specified columns
    data.drop(columns=['loan_amnt', 'loan_status', 'loan_percent_income', 'loan_grade'], inplace=True)
    #average life expectancy in uk is 81 in the UK
    data = data[data['person_age'] < 81]
    #average retirement age is 66 currently, filtering employment length
    emp_max = 66 - 18
    data = data[data['person_emp_length'] <= emp_max]
    #since the application gives out personal loans, only want to keep instances of loans which are classified as 'personal'
    data = data[~data['loan_intent'].isin(['EDUCATION', 'MEDICAL', 'VENTURE', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])]
    data = data[data['person_home_ownership'] != 'OTHER']
     #'other' is hard to quantify when gathering user information
    print('Dataset has been cleaned. Current shape:', data.shape)
    print('Amount of null values in each column is:')
    print(data.isnull().sum())      
    return data


#function to transform the data
def transform_data(data):
    #renaming columns
    data = data.rename(columns= {"person_age": "Age", "person_income": "Income", "person_emp_length": "Employment Years", "cb_person_cred_hist_length": "Credit History Years",
    "person_home_ownership": "Home Ownership", "cb_person_default_on_file": "Defaulted on Loan"})
    #encoding categorical columns for model and EDA
    label_encoder = LabelEncoder()
    data['HomeOwnership_encoded'] = label_encoder.fit_transform(data['Home Ownership'])
    data['Defaulted_encoded'] = label_encoder.fit_transform(data['Defaulted on Loan'])
    print('Data transformation has been completed.')
    return data

#function to split the dataset for model building and to perform random oversampling
def split_data(data, feature_columns, target_column):
    X = data[feature_columns]
    y = data[target_column]
    ros = RandomOverSampler(sampling_strategy=1)
    X_res, y_res = ros.fit_resample(X, y)
    print('Data has been split.')
    return train_test_split(X_res, y_res, test_size=0.2, random_state=42)

#function to train the model 
def train_model(X_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    print('Your model has been successfully trained')
    return rf_classifier

#function to evalute the model's performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

#function to cross validate the model
def cross_validate(model, X, y):
    cv = StratifiedKFold(n_splits=5)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print("Cross-Validation Scores:", scores)
    print("Average score:", np.mean(scores))


#function to pickle file for deployment
def save_model(model, file_path):
   with open(file_path, 'wb') as file:
      pickle.dump(model, file)
   print("Model saved successfully.")

#function to calculate the shap values - based on defaulted class (1)
def calculate_shap_values(model, X):
    explainer = fasttreeshap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_values_array = np.array(shap_values)
    print('The shapley values have been created')
    return shap_values_array

def shap_percentages(shap_values_array, feature_columns):
    shap_default = shap_values_array[1, :, :]
    av_shapvalues = np.mean(np.abs(shap_default), axis=0)
    total_shapvalues = np.sum(av_shapvalues)

    print("Feature contributions for defaulting on a loan:")
    for i, feature in enumerate(feature_columns):
        contribution_percent = (av_shapvalues[i] / total_shapvalues) * 100
        print(f"{feature} contributed to {contribution_percent:.2f}% of the decision")
        print('The process has been completed')

def main():
    #load the data
    data = load_data("dataanalysis/creditriskdataset.csv")

    #clean the data
    data = clean_data(data)

    #transform the data
    data = transform_data(data)

    #split the data and random oversample
    feature_columns = ['Age', 'Income', 'HomeOwnership_encoded', 'Employment Years', 'Credit History Years']
    target_column = 'Defaulted_encoded'
    X_train, X_test, y_train, y_test = split_data(data, feature_columns, target_column)

    #train the model
    model = train_model(X_train, y_train)

    #evaluate the model
    evaluate_model(model, X_test, y_test)

    #cross validate the model
    cross_validate(model, data[feature_columns], data[target_column])

    #save the model as a pickle file
    save_model(model, "defaulted_loan_model.pkl")

    #calculate the shap values
    shap_values_array = calculate_shap_values(model, X_test)

    #calculate the contributions to the model - based on defaulted class (1)
    shap_percentages(shap_values_array, feature_columns)


#runs the script
if __name__ == "__main__":
    main()

