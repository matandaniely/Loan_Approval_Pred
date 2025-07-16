🏦 # Loan Approval Classification Project

This project presents a full pipeline for building, evaluating, and comparing classification models to predict loan approval outcomes based on applicant information. The dataset, sourced from a public loan dataset, is analyzed using data cleaning, visualization, feature engineering, and supervised machine learning algorithms.

📁 Project Structure

Loan_Classification_and_Predictions.ipynb: Jupyter Notebook containing all steps from data preprocessing to model evaluation.
loan approval classification external dataset.csv: Dataset used in the project.    

🧠 Problem Statement

The objective is to predict whether a loan will be approved (Y) or denied (N) based on multiple applicant attributes such as income, credit history, employment status, and more. This is framed as a binary classification task.

📊 Dataset Overview

The dataset includes 304 entries with the following features:

Feature	Type	Description    
Gender	Categorical	Applicant gender    
Married	Categorical	Marital status    
Dependents	Categorical	Number of dependents    
Education	Categorical	Graduate/Not Graduate    
Self_Employed	Categorical	Employment type    
ApplicantIncome	Numeric	Monthly income of the applicant    
CoapplicantIncome	Numeric	Monthly income of co-applicant    
LoanAmount	Numeric	Loan amount requested    
Loan_Amount_Term	Numeric	Term of the loan    
Credit_History	Numeric	Credit history (1 = Good, 0 = Bad)    
Property_Area	Categorical	Urban/Rural/Semiurban area    
Loan_Status	Target	Loan approved (Y/N)    

🧹 Data Cleaning & Preprocessing    
    
Handling Missing Values:    
Mean imputation for numerical columns    
Mode imputation for categorical columns   
    
Encoding Categorical Variables:    
One-hot encoding applied to convert non-numeric columns into machine-readable form    
    
Feature Scaling:    
StandardScaler used to normalize numerical features    
    
Train-Test Split:    
70% training and 30% testing split    
    
⚙️ Models Implemented    
    
Model	Description    
Logistic Regression	Baseline linear classifier    
Random Forest Classifier	Ensemble method using multiple decision trees    
Performance metrics like accuracy, confusion matrix, and classification report were used to evaluate both models.    
    
📈 Visualizations    
    
Count plots and heatmaps using Seaborn for feature distributions and correlations.    
ROC curves and model comparison charts.    
    
🧪 Sample Code Snippets    
    
🔍 Data Exploration    
loan.isnull().sum()    
loan.describe()    
    
🚀 Model Training    
lr = LogisticRegression()    
lr.fit(X_train, y_train)    
    
rf = RandomForestClassifier()    
rf.fit(X_train, y_train)    

📊 Evaluation    
from sklearn.metrics import classification_report    
print(classification_report(y_test, rf.predict(X_test)))    
    
💡 Key Insights    
    
Credit history was the most influential factor in predicting loan approval.    
Logistic Regression performed well as a baseline, while Random Forest achieved higher accuracy.    
Data imbalance between approved (Y) and denied (N) loans may require further sampling techniques for improvement.    
    
📌 Future Improvements    
    
Implement cross-validation and grid search for hyperparameter tuning.    
Explore XGBoost or LightGBM for improved performance.    
Address class imbalance using SMOTE or undersampling techniques.    
    
🧑‍💻 Author    

Matan Joel Daniely
MSc in Fintech & Business Analytics – EADA Business School, Barcelona
Focused on combining business logic with technical implementation through Python, SQL, and data science.
