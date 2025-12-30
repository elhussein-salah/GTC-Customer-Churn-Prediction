import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {background-color: #f7fafd;}
    .stButton>button {font-size: 18px;}
    .css-1cpxqw2 {font-size:18px;}
    .css-1d391kg {font-size:18px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📉 Telecom Customer Churn Prediction")
st.markdown(
    """
    Welcome to the **GTC Churn Prediction Project Web App**!

    This interactive tool predicts whether a telecom customer is likely to churn based on their service usage and demographic data.

    ---
    """
)

@st.cache_resource(show_spinner="Loading model…")
def load_model_and_encoders():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

def preprocess_input(df, encoders):
    df = df.copy()
    # Drop columns not in model training
    required_features = [
        'gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService',
        'MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
        'TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling',
        'PaymentMethod','MonthlyCharges','TotalCharges'
    ]
    # Keep only columns used for training (ignore extra columns like 'customerID')
    df = df[[col for col in required_features if col in df.columns]]

    # Convert columns to numeric as before
    for col in ['TotalCharges', 'MonthlyCharges', 'tenure']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    df['MonthlyCharges'] = df['MonthlyCharges'].fillna(0)
    df['tenure'] = df['tenure'].fillna(0)

    # Feature engineering...
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-12', '12-24', '24-48', '48-72'])
    df['ChargesRatio'] = np.where(df['MonthlyCharges'] > 0, df['TotalCharges'] / df['MonthlyCharges'], 0)
    df['AvgMonthlySpend'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], 0)

    for column, encoder in encoders.items():
        df[column] = df[column].astype(str)
        known_classes = set(encoder.classes_)
        df[column] = df[column].apply(lambda x: x if x in known_classes else encoder.classes_[0])
        df[column] = encoder.transform(df[column])
    return df

def predict_churn(df, model, encoders):
    X = preprocess_input(df, encoders)
    if "Churn" in X.columns:
        X = X.drop(columns=["Churn"])
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return y_pred, y_prob, X

def plot_shap_summary(model, X, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10, 4))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

def plot_shap_waterfall(model, X_row, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)
    # For binary classification, shap_values and expected_value are lists/arrays
    if isinstance(shap_values, list) or (hasattr(shap_values, "shape") and len(shap_values.shape) == 3):
        # Use index 1 for the positive class ("churned")
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        shap_value = shap_values[1][0]
    else:
        expected_value = explainer.expected_value
        shap_value = shap_values[0]
    plt.figure(figsize=(8, 3))
    shap.plots._waterfall.waterfall_legacy(
        expected_value, shap_value, X_row.iloc[0], feature_names=feature_names, show=False
    )
    st.pyplot(plt.gcf())
    plt.clf()

# Sidebar: About & Template
with st.sidebar:
    st.header("About This App")
    st.markdown(
        """
        - **Team:** [Elsayed Ashraf Bakry From Team 13]
        - **Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)
        - **Project:** Telecom Customer Churn Prediction (GTC Internship Final Project)

        ---
        """
    )
    st.subheader("Input Template")
    if st.button("Download CSV Template"):
        template_path = "input_template.csv"
        if not os.path.exists(template_path):
            columns = [
                'gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService',
                'MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                'TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling',
                'PaymentMethod','MonthlyCharges','TotalCharges'
            ]
            sample_data = [[
                "Female", 0, "Yes", "No", 12, "Yes", "No", "DSL", "Yes", "No", "Yes", "No", "Yes",
                "No", "Month-to-month", "Yes", "Electronic check", 29.85, 350.0
            ]]
            pd.DataFrame(sample_data, columns=columns).to_csv(template_path, index=False)
        with open(template_path, "rb") as f:
            st.download_button("Download Template CSV", f, file_name="input_template.csv", mime="text/csv")

# Load model and encoders
try:
    model, encoders = load_model_and_encoders()
except Exception as e:
    st.error("Model or encoders file not found. Please ensure 'model.pkl' and 'encoders.pkl' are present.")
    st.stop()

st.header("Provide Customer Data")

input_method = st.radio(
    "How would you like to provide data?",
    ("Upload CSV file", "Enter features manually")
)

expected_features = [
    'gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService',
    'MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
    'TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling',
    'PaymentMethod','MonthlyCharges','TotalCharges'
]

if input_method == "Upload CSV file":
    uploaded_file = st.file_uploader("Choose your CSV file", type=["csv"])
    if uploaded_file:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.write("Preview of input data:")
            st.dataframe(df_input.head())
            missing_cols = [c for c in expected_features if c not in df_input.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                if st.button("Predict on Uploaded Data", type="primary"):
                    y_pred, y_prob, X = predict_churn(df_input, model, encoders)
                    results = df_input.copy()
                    results["Churn_Prediction"] = np.where(y_pred == 1, "Yes", "No")
                    if y_prob is not None:
                        results["Churn_Probability"] = np.round(y_prob, 3)
                    st.subheader("Prediction Results")
                    st.dataframe(results[["Churn_Prediction", "Churn_Probability"]].head(20))
                    st.download_button(
                        label="Download Results as CSV",
                        data=results.to_csv(index=False),
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                    # SHAP summary for batch upload
                    st.markdown("---")
                    st.subheader("Global Model Explanation (SHAP Summary)")
                    st.info("The plot below visualizes the most important features for churn prediction across your uploaded customers.")
                    plot_shap_summary(model, X, X.columns.tolist())
        except Exception as e:
            st.error(f"Error processing file: {e}")

else:
    st.markdown("**Enter customer features below:**")
    with st.form("manual_entry_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
            Partner = st.selectbox("Partner", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=1)
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
            MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        with c2:
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        with c3:
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=0.0, step=0.01)
            TotalCharges = st.number_input("Total Charges", min_value=0.0, value=0.0, step=0.01)
        submitted = st.form_submit_button("Predict Churn (Manual Entry)", type="primary")

    if submitted:
        manual_df = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }])
        y_pred, y_prob, X_man = predict_churn(manual_df, model, encoders)
        st.subheader("Prediction Result")
        st.success(
            f"Prediction: **{'Churn' if y_pred[0]==1 else 'No Churn'}**  "
            f"({100*y_prob[0]:.1f}% probability of churn)"
        )
        # Detailed SHAP for this prediction
        st.markdown("---")
        st.subheader("Model Explanation for This Customer (SHAP)")
        st.info("See which features most influenced this prediction below.")
        try:
            plot_shap_waterfall(model, X_man, X_man.columns.tolist())
        except Exception as e:
            st.warning(f"Could not generate SHAP waterfall plot: {e}")

st.markdown("---")
with st.expander("🔎 Example Feature Importances (from training)"):
    feature_importance_sample = {
        'Contract': 0.32, 'MonthlyCharges': 0.25, 'tenure': 0.21,
        'InternetService': 0.10, 'TechSupport': 0.07, 'TotalCharges': 0.05
    }
    st.bar_chart(pd.Series(feature_importance_sample).sort_values(ascending=False))

with st.expander("📊 Example: Churn Distribution in Training Data"):
    # Example plot (replace with real EDA if desired)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4,2))
    churn_counts = pd.Series({"No": 5163, "Yes": 1869})
    churn_counts.plot(kind='bar', color=["#1f77b4", "#ff7f0e"], ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Churn Distribution")
    st.pyplot(fig)

st.markdown(
    """
    ---
    <div style='text-align:center'>
    <small>
    Made with ❤️ by Elsayed Ashraf Bakry &nbsp;|&nbsp;
    <a href="https://github.com/elhussein-salah/GTC-Customer-Churn-Prediction" target="_blank">Project GitHub</a>
    </small>
    </div>
    """,
    unsafe_allow_html=True
)