#Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import pickle

# Loading the dataset and drop unnecessary columns
# Set the path to your CSV file here. Example: 'PROJ/WA_Fn-UseC_-Telco-Customer-Churn.csv'
csv_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(csv_path)
df = df.drop(['customerID'], axis=1)

#Converting 'TotalCharges' to numeric, forcing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#Dropping rows with NaN values
df.dropna(inplace=True)

#Insights:
#1. Customer ID removed as it is not required for modelling
#2. No missing values in the dataset
#3. Missing values in the TotalCharges column were replaced with 0
#4. Class imbalance identified in the target

#EDA
#Feature engineering
# Tenure groups
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-12', '12-24', '24-48', '48-72'])

# Charges ratio (avoid division by zero)
df['ChargesRatio'] = np.where(df['MonthlyCharges'] > 0, df['TotalCharges'] / df['MonthlyCharges'], 0)

# Average monthly spend
df['AvgMonthlySpend'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], 0)

#Label encoding of target column
df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0}).infer_objects()

# Identify categorical columns (object or category dtype)
object_columns = df.select_dtypes(include=["object", "category"]).columns

# Initialize dictionary to save encoders
encoders = {}


# Apply Label Encoding
for column in object_columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column].astype(str))  # ensure string type
    encoders[column] = label_encoder

# Save the encoders for future use (important for deployment phase)
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

encoders

# splitting the features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addressing class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

#Training multiple models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)   # training on balanced data
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

results_df = pd.DataFrame(results).T
print(results_df.sort_values(by="ROC-AUC", ascending=False))

# Hyperparameter tuning for Random Forest
param_dist_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42, class_weight="balanced")

random_search_rf = RandomizedSearchCV(
    rf, param_distributions=param_dist_rf,
    n_iter=20, cv=3, scoring='accuracy',
    random_state=42, n_jobs=-1, verbose=2
)

random_search_rf.fit(X_train_smote, y_train_smote)

#Hyperparameter tuning for XGBoost
param_dist_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

random_search_xgb = RandomizedSearchCV(
    xgb, param_distributions=param_dist_xgb,
    n_iter=20, cv=3, scoring='accuracy',
    random_state=42, n_jobs=-1, verbose=2
)

random_search_xgb.fit(X_train_smote, y_train_smote)

#re-evaluating the tuned models
# Evaluate tuned models on test set
best_rf = random_search_rf.best_estimator_
best_xgb = random_search_xgb.best_estimator_

for name, model in {"Tuned Random Forest": best_rf, "Tuned XGBoost": best_xgb}.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Test Accuracy: {acc:.4f}")



# Save the best trained model as a pickle file (choose best_xgb or best_rf)
import joblib
joblib.dump(model, "model.pkl", compress=3)  # use compress=3 