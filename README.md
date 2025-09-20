# **Customer Churn Prediction for a Telecom Company**

This project presents a complete machine learning pipeline for predicting customer churn for a telecom company. The primary goal is to build a robust predictive model that can identify customers at risk of leaving the service, allowing the business to implement targeted retention strategies.

-----

### **Project Overview**

The project is developed as a Jupyter Notebook (`customer_churn_prediction_gtc (1).ipynb`) that follows a standard machine learning workflow.

### **1. Data Preparation**

This initial phase focuses on cleaning and preparing the raw data for analysis. The key steps performed were:

  * **Loading and Initial Inspection**: The `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset was loaded, and initial checks were performed to understand its structure, including the number of rows and columns.
  * **Handling Missing Values**: The `TotalCharges` column, which was incorrectly loaded as an object type, was converted to a numeric format. During this process, 11 rows with missing values were identified and dropped, as this was a small fraction of the total data.
  * **Data Cleaning**: The `customerID` column was removed as it does not contribute to the predictive model.

-----

### **2. Exploratory Data Analysis (EDA)**

In this section, the cleaned data was explored to gain a deeper understanding of the features and their relationships with the target variable (`Churn`). The following analyses were performed:

  * **Target Variable Analysis**: The distribution of the `Churn` variable was examined, revealing a significant class imbalance (73.5% 'No' churn vs. 26.5% 'Yes' churn). This insight is critical for selecting appropriate modeling techniques.
  * **Feature-Target Relationship**: Visualizations were used to analyze how different features, such as `Contract`, `InternetService`, `MonthlyCharges`, and `tenure`, correlate with the churn rate. This provides a foundational understanding for feature selection and engineering.

-----

### **3. Feature Engineering & Preprocessing**

To prepare the data for the machine learning models, the following steps were undertaken:

  * **Categorical Encoding**: Categorical features were converted into a numerical format suitable for machine learning algorithms.
  * **Feature Scaling**: Numerical features were scaled to ensure they have a similar range, preventing any single feature from dominating the model due to its magnitude.
  * **Data Splitting**: The dataset was split into training and testing sets to evaluate the model's performance on unseen data.
  * **Handling Class Imbalance**: To address the imbalanced `Churn` variable, the **Synthetic Minority Oversampling Technique (SMOTE)** was applied to the training data. This technique generates synthetic examples for the minority class, helping the model learn the patterns of churn more effectively.

-----
