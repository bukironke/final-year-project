# Credit Scoring Tool

## Information about this repository
Please note that this is cloned from my university Gitlab
About:
My project is to develop a web application that can successfully determine a user's credit score based on information given by the user.
My aim is to develop an interactive and professional UI experience for the user. The assessment of the user's score will be implemented
with a machine learning model which will gather patterns and results from a dataset and then apply that to the assessment of the user's information.
Below are the folders in my project, with relevant information

## Folders
**'dataanalysis'**
- Contains the CSV file the models were trained on, the link to the dataset is as followed: 
https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data
- 'mlmodel.ipynb' - my first attempt at building a machine learning model - was more for trial and error purposes but I have kept it there as evidence of my evolution during this project
- 'mlmodel2.ipynb' - used to build the classification model (whether the user will default on a loan or not) - consists of the data cleaning, transformation, EDA, training and testing with different algorithms, cross validation, SHAPLEY values, then pickling the model for deployment
- 'mlmodel3. ipynb' - used to build the regression model (predicting the user's interest rate) - similar to mlmodel2.ipynb - consists of data cleaning, transformation, EDA, training and testing with different algorithms and then pickling the model for deployment
- Upon request by Rajeez, I have created two Python files which hold the raw code for both models with no EDA or visualisation for simplicitiy purposes called classification_model.py (mlmodel2) and regression_model.py (mlmodel3).

**'interimreport'**
- consists of the interim report submitted in November 2023

**'models'**
- consists of the classification (defaulted_loan_model.pkl) and regression model (interest_rate_model.pkl) pickled for deployment

**'scrumboard'**
- consists of screenshots of the proposal timelines for certain tasks to be completed in relation to this project

**'static'**
- consists of all the img, javascript and css files within the application

**'templates'**
- consists of all the html files for all the pages within the application

**'app.py'**
- with the use of Flask, is responsible for defining the app's routes


## Source Codes from Online

Templates found online were used entirely or adapted to fit the requirements of the project

**THE LOGIN PAGE**
- /static/js/loginpage.js 
- /static/css/loginPage.scss
- /static/css/loginPage.css
- /templates/loginPage.html

**LINK:**
https://codepen.io/elisandromoreira/pen/wvavgxr

**THE RESULTS/DASHBOARD PAGE**
- /static/js/creditscoreresult.js
- /static/css/creditscoreResult.css
- /templates/creditscoreResult.html

**GOOGLE CHARTS**
- PIE CHART: https://developers.google.com/chart/interactive/docs/gallery/piechart
- LINE CHART: https://developers.google.com/chart/interactive/docs/gallery/linechart

**LINK**
https://codepen.io/thejamesnash/pen/rRKYJv


## Languages and libraries used

**Programming Languages used**
- Python
- HTML 
- CSS 
- SCSS
- Javascript


**Libraries and frameworks used**
- Flask
- Numpy
- Pandas
- SHAP 
- Fasttreeshap (https://github.com/linkedin/FastTreeSHAP/tree/master)
- Scikit-Learn
- Pdfkit 
- JSON 
- Imblearn
- Pickle 
- Matplotlib
- Seaborn 

For Pdfkit to work, I had to install WKHTMLTOPDF onto my local machine (https://wkhtmltopdf.org/)

Most frameworks used were already installed onto my machine, however for this project, I installed SHAP, FastTreeSHAP, Imblearn, Pickle, PDFKIT via PIP. 
