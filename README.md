***

# Customer Churn Prediction for a Telecom Company 📉

This project develops a predictive model and user-friendly web application to identify customers at risk of churning from a telecom service. Leveraging a comprehensive machine learning pipeline and a professional UI powered by Streamlit, the solution provides actionable insights and visualizations for both data scientists and business stakeholders.

The entire data science workflow is documented within the `customer_churn_prediction_gtc.ipynb` Jupyter Notebook. The deployment-ready app is implemented in `app.py`.

---

## **1. Data Preparation & Cleaning**

This phase focused on transforming raw data into a clean, structured format suitable for analysis and modeling.

- **Data Loading:** The `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset was loaded and inspected for structure and integrity.
- **Handling Missing Values:** The `TotalCharges` column was cleaned by converting to numeric type. 11 rows with missing values were removed, as they represented a negligible portion of the dataset.
- **Irrelevant Feature Removal:** The `customerID` column was dropped, as it has no predictive value and can cause issues during model deployment.

---

## **2. Exploratory Data Analysis (EDA) & Feature Engineering**

This phase explored the data and engineered new features to improve model performance and business interpretability.

- **Class Imbalance:** The `Churn` variable showed significant imbalance (73.5% No vs. 26.5% Yes). This guided the modeling strategy and evaluation metrics.
- **Feature-Target Relationships:** Visualizations were created to understand how variables like `tenure`, `Contract`, `MonthlyCharges`, and `InternetService` affect churn.
- **Advanced Feature Creation:**
    - **Tenure Grouping:** Customers were segmented into `TenureGroup` buckets (e.g., short-term, long-term) to capture non-linear patterns.
    - **ServiceCount:** Aggregated the number of services a customer subscribes to, measuring engagement.
    - **Charge Ratios:** Created `ChargesRatio` and `AvgMonthlySpend` to capture customer spending habits.

---

## **3. Model Training, Evaluation & Interpretability**

Covers the end-to-end machine learning workflow from preprocessing to explainable predictions.

- **Data Preprocessing:** Categorical features were encoded, numerical features scaled, and the data split into training and test sets.
- **Class Imbalance Handling:** Used **SMOTE** (Synthetic Minority Oversampling Technique) to balance the classes in training data and avoid model bias.
- **Model Selection & Tuning:** Trained and optimized both **Random Forest** and **XGBoost** classifiers via `RandomizedSearchCV`.
- **Robust Evaluation:** Applied **5-fold cross-validation** for reliable performance metrics.
- **Model Interpretability:**
    - **Feature Importance:** Used built-in and SHAP-based global feature importances.
    - **SHAP Explanations:** Provided local (per-customer) and global (dataset-wide) SHAP plots for full transparency.

---

## **4. Streamlit App: Professional UI & Visualizations**

A modern, interactive web app (`app.py`) was built for business users and data scientists:

- **Input Options:** Upload a CSV file or manually enter customer features.
- **Visualizations:**
    - **SHAP Summary Plot:** Shows the top features influencing churn across all uploaded customers.
    - **SHAP Waterfall Plot:** Explains the prediction for each individual customer.
    - **Feature Importance & EDA Charts:** Bar plots and distributions for key features and churn outcomes.
- **Downloadable Results:** Users can download churn predictions for their uploaded data.
- **Dark Mode:** The app supports dark mode via Streamlit's theme configuration.
- **Performance:** The model is compressed for fast loading and efficient deployment.

---

## **5. Results**

The project successfully delivered a robust, interpretable churn prediction solution:

- **Performance:** Both XGBoost and Random Forest achieved strong predictive results.
- **Business Insights:** The app highlights actionable drivers of churn and empowers targeted retention efforts.
- **Deployment:** The streamlined, professional UI enables easy adoption by non-technical users.

---

## **6. How to Run the App**

1. **Run the app:**
   ```bash
   streamlit run app.py
   ```
2. **Interact:**
   - Upload your customer data (CSV) or enter features manually.
   - View predictions and SHAP-based visualizations.
   - Download prediction results.

---

## **7. Repository Structure**

```
.
├── app.py                       # Streamlit web app
├── customer_churn_prediction_gtc.ipynb  # Full data science notebook
├── model.pkl                    # Compressed trained model (joblib)
├── encoders.pkl                 # Categorical feature encoders
├── WA_Fn-UseC_-Telco-Customer-Churn.csv            # Dataset
└── README.md                    # Project documentation
```

---

## **8. Credits**

- **Team:** [13]
- **Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Organization:** GTC Internship

---

**For questions or contributions, please [open an issue](https://github.com/elhussein-salah/GTC-Customer-Churn-Prediction/issues) or submit a pull request.**

***
