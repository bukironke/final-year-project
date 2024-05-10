#importing relevant libraries
#Please note that most of the EDA and trial and error is in the notebook scripts (mlmodel1, 2 and 3), this is just to give a more simplistic illustration of my model
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
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

#function to perform data transformation progress
def transform_data(data):
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


#function to split the dataset for model building 
def split_data(data, feature_columns, target_column):
    X = data[feature_columns]
    y = data[target_column]
    print('Data has been split.')
    return train_test_split(X, y, test_size=0.2, random_state=42)

#function to train the model 
def train_model(X_train, y_train):
    best_regressor = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=2, min_samples_split=10, max_features='log2', random_state=0, oob_score=True)
    best_regressor.fit(X_train, y_train)
    print('Your model has been successfully trained')
    return best_regressor

#function to evalute the model's performance
def evaluate_model(best_regressor, X_test, X_train, y_test, y_train):

    predictions_train = best_regressor.predict(X_train)

    oob_score = best_regressor.oob_score_
    print(f'Out-of-Bag Score: {oob_score}')

    mse_train = mean_squared_error(y_train, predictions_train)
    print(f'Mean Squared Error (Training): {mse_train}')

    r2_train = r2_score(y_train, predictions_train)
    print(f'R-squared (Training): {r2_train}')

    predictions_test = best_regressor.predict(X_test)

    mse_test = mean_squared_error(y_test, predictions_test)
    print(f'Mean Squared Error (Test): {mse_test}')

    r2_test = r2_score(y_test, predictions_test)
    print(f'R-squared (Test): {r2_test}')
    print('The process has now been completed')

#function to pickle file for deployment
def save_model(model, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    print("Model saved successfully.")

def main():
    #load the data
    data = load_data("dataanalysis/creditriskdataset.csv")

    #clean the data
    data = clean_data(data)

    #transform the data
    data = transform_data(data)

    #split the data and random oversample
    feature_columns = ['Age', 'Income', 'Employment Years', 'HomeOwnership_encoded', 'Defaulted_encoded']
    target_column = 'Interest Rate'
    X_train, X_test, y_train, y_test = split_data(data, feature_columns, target_column)

    #train the model
    model = train_model(X_train, y_train)

    #evaluate the model
    evaluate_model(model, X_test, X_train, y_test, y_train)

    #save the model as a pickle file
    save_model(model, "interest_rate_model.pkl")

#runs the script
if __name__ == "__main__":
    main()
