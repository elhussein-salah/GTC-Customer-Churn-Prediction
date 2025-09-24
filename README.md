***

# Customer Churn Prediction for a Telecom Company 📉

This project develops a predictive model to identify customers at risk of churning from a telecom service. By leveraging a comprehensive machine learning pipeline, the goal is to provide actionable insights for implementing targeted customer retention strategies.

The entire project is documented within the `customer_churn_prediction_gtc.ipynb` Jupyter Notebook.

---

### **1. Data Preparation & Cleaning**

This phase focused on transforming raw data into a clean, structured format suitable for analysis.
- **Data Loading:** The `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset was loaded and inspected for its structure and integrity.
- **Handling Missing Values:** The `TotalCharges` column was cleaned by converting it to a numeric type, and 11 rows with missing values were removed, as they represented a negligible portion of the data.
- **Irrelevant Feature Removal:** The `customerID` column was dropped, as it has no predictive value.

---

### **2. Exploratory Data Analysis (EDA) & Feature Engineering**

In this crucial phase, the data was explored to uncover patterns and relationships. New features were engineered to improve model performance.
- **Class Imbalance:** Analysis of the `Churn` variable revealed a significant imbalance (73.5% No vs. 26.5% Yes), which informed the choice of modeling techniques.
- **Feature-Target Relationships:** Visualizations were used to understand how factors like `tenure`, `Contract`, `MonthlyCharges`, and `InternetService` influence churn rates.
- **Advanced Feature Creation:** To deepen the analysis, several new features were engineered:
    - **Tenure Grouping:** Customers were segmented into `TenureGroup` buckets (e.g., short-term, long-term) to capture non-linear relationships.
    - **Usage Intensity Score:** A `ServiceCount` feature was created by aggregating the number of services a customer uses, providing a measure of their engagement.
    - **Charge Ratios:** Features like `ChargesRatio` and `AvgMonthlySpend` were derived to capture customer spending habits.

---

### **3. Model Training, Evaluation & Interpretability**

This section covers the core machine learning workflow, from model selection to final evaluation and explanation.
- **Data Preprocessing:** Categorical features were encoded, and numerical features were scaled to standardize their ranges. The dataset was then split into training and testing sets.
- **Addressing Class Imbalance:** The **Synthetic Minority Oversampling Technique (SMOTE)** was applied to the training data to balance the classes and prevent the model from becoming biased toward the majority class.
- **Model Selection & Tuning:** A **Random Forest Classifier** and an **XGBoost Classifier** were selected and optimized using `RandomizedSearchCV` for hyperparameter tuning.
- **Robust Evaluation:** The best-performing models were evaluated using **5-fold cross-validation** to ensure the performance metrics were robust and not dependent on a single data split.
- **Model Interpretability:** To provide actionable business insights, the model's decisions were explained using:
    - **Feature Importance Plots** to identify the most influential features globally.
    - **SHAP (SHapley Additive exPlanations) plots** to explain how individual features contribute to a specific customer's churn prediction.

---

### **4. Results**

The project successfully delivered a robust churn-prediction model, with the XGBoost Classifier and Random Forest model achieving a strong predictive performance. The model’s interpretability features allow for a clear understanding of the key drivers of customer churn, enabling business teams to design effective retention strategies.
