**Loan Approval Classification Project**

This project presents a full pipeline for building, evaluating, and comparing classification models to predict loan approval outcomes based on applicant information. The dataset, sourced from a public loan dataset, is analyzed using data cleaning, visualization, feature engineering, and supervised machine learning algorithms.


**Why is it important to test two prediction models?**    
1. No One-Size-Fits-All Model    
Different models make different assumptions and behave differently depending on the nature of the dataset. For example:    
Logistic Regression assumes linear relationships.    
Random Forest can capture nonlinear interactions and is more flexible.    
2. Check for Overfitting vs. Generalization    
One model might perform extremely well on training data but poorly on unseen data (overfitting).    
Testing multiple models helps find the one that generalizes best to real-world data.    
3. Baseline vs. Advanced     
A simple model like Logistic Regression is often used as a baseline.     
Comparing it to a more complex model like Random Forest helps understand whether the added complexity brings real improvement.    


**Project Structure**

- Loan_Classification_and_Predictions.ipynb: Jupyter Notebook containing all steps from data preprocessing to model evaluation.    
- loan approval classification external dataset.csv: Dataset used in the project.     

**Problem Statement**

The objective is to predict whether a loan will be approved (Y) or denied (N) based on multiple applicant attributes such as income, credit history, employment status, and more. This is framed as a binary classification task.

**Dataset Overview**

The dataset includes 838 datapoints with the following features:

| Feature            | Type        | Description                        |
| ------------------ | ----------- | ---------------------------------- |
| Gender             | Categorical | Applicant gender                   |
| Married            | Categorical | Marital status                     |
| Dependents         | Categorical | Number of dependents               |
| Education          | Categorical | Graduate/Not Graduate              |
| Self\_Employed     | Categorical | Employment type                    |
| ApplicantIncome    | Numeric     | Monthly income of the applicant    |
| CoapplicantIncome  | Numeric     | Monthly income of co-applicant     |
| LoanAmount         | Numeric     | Loan amount requested              |
| Loan\_Amount\_Term | Numeric     | Term of the loan                   |
| Credit\_History    | Numeric     | Credit history (1 = Good, 0 = Bad) |
| Property\_Area     | Categorical | Urban/Rural/Semiurban area         |
| Loan\_Status       | Target      | Loan approved (Y/N)                |
  
    
**Data Cleaning & Preprocessing**    
    
Handling Missing Values:    
Mean imputation for numerical columns    
Mode imputation for categorical columns   
    
Encoding Categorical Variables:    
One-hot encoding applied to convert non-numeric columns into machine-readable form    
    
Feature Scaling:    
StandardScaler used to normalize numerical features    
    
Train-Test Split:    
70% training and 30% testing split    
    
**Models Implemented**    
    
Model	Description    
Logistic Regression	Baseline linear classifier    
Random Forest Classifier	Ensemble method using multiple decision trees    
Performance metrics like accuracy, confusion matrix, and classification report were used to evaluate both models.    
    
**Visualizations**   
    
Count plots and heatmaps using Seaborn for feature distributions and correlations.    
ROC curves and model comparison charts.    
    
**Sample Code Snippets**   

    The data couldn't be used raw:    
    1. Data cleaning and imputation was needed     
    2. Creating dummies in order to change categorical information into numerical data points      
    3. We scale features in machine learning to make sure all numeric inputs are on a similar scale, which improves the performance and reliability of many algorithms.    
    4. Creating a correlation heatmap to identify whice features have an impact on results - which is crucial for a logistic regression model so we could eliminate noise (for more complex ML models it is less necessary to eliminate 0 correlated featuess).     


The result of creating dummies:     
<img width="872" height="375" alt="The dummies" src="https://github.com/user-attachments/assets/7b5554d9-0ef8-4e7a-9e7a-9961fc2d5698" />

    The Correlation Matrix betwen the features, their dummies and scaled datapoints      
      
<img width="1362" height="1156" alt="Feature Corr" src="https://github.com/user-attachments/assets/4c3ae97c-e1fa-4ccf-a469-9fe9883b487f" />
        

    
**Data Exploration**    
loan.isnull().sum()    
<img width="854" height="416" alt="Exploration1" src="https://github.com/user-attachments/assets/2df97cfa-12de-49c5-bd10-268b25348169" />    
    
loan.info()    
<img width="962" height="597" alt="loan info()" src="https://github.com/user-attachments/assets/e210b48c-2c43-428a-9cac-d9882be47393" />    
    
    
**Model Training**    
lr = LogisticRegression()    
lr.fit(X_train, y_train)    

<img width="1236" height="801" alt="LogReg" src="https://github.com/user-attachments/assets/d321fe61-a820-49a6-b8dd-b37dc0ca0c04" />
<img width="1247" height="768" alt="LogReg Pred" src="https://github.com/user-attachments/assets/cb1390cd-92ac-4691-9de3-65e447531e47" />

    
rf = RandomForestClassifier()    
rf.fit(X_train, y_train)    
    
<img width="1252" height="755" alt="RndF" src="https://github.com/user-attachments/assets/ab2adc94-7e50-42ee-a2d2-2a077cac65e9" />


**Evaluation**    
print("Logistic Regression:")
print("MSE:", mean_squared_error(y_test, y_pred_logreg))
print("R2 Score:", r2_score(y_test, y_pred_logreg))
print()

print("Random Forest Regressor:")
print("MSE:", mean_squared_error(y_test, y_pred_rnd_clf))
print("R2 Score:", r2_score(y_test, y_pred_rnd_clf))   


<img width="1261" height="561" alt="n_estimators and evaluation" src="https://github.com/user-attachments/assets/ea067902-c20e-4c8c-b6b7-c2daa22a5fea" />

**Random Forest Prediction Results**

<img width="996" height="547" alt="output" src="https://github.com/user-attachments/assets/b9e9da8f-a03e-49e2-8e26-10c41875a0ce" />


**Key Insights**    
    
Credit history was the most influential factor in predicting loan approval.    
Logistic Regression performed well as a baseline, while Random Forest achieved higher accuracy.    
Data imbalance between approved (Y) and denied (N) loans may require further sampling techniques for improvement.    
    
**Future Improvements**    
    
Implement cross-validation and grid search for hyperparameter tuning.    
Explore XGBoost or LightGBM for improved performance.    
Address class imbalance using SMOTE or undersampling techniques.    
    
**Author**    

Matan Joel Daniely
MSc in Fintech & Business Analytics – EADA Business School, Barcelona
Focused on combining business logic with technical implementation through Python, SQL, and data science.
