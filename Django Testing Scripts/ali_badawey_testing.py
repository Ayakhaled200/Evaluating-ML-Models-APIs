import os
import sys
import django
import httpx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import shap
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up the Django environment
django_project_path = r'E:\Giza Systems\Model Evaluation\Ali_Badawy\pythonProject1\Django_API'
sys.path.append(django_project_path)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myeval_api.settings')

try:
    django.setup()
except Exception as e:
    print(f"Django setup failed: {e}")

# Load the models and scalers used in Django's views.py for preprocessing
model_logreg_path = r'E:\Giza Systems\Model Evaluation\Ali_Badawy\pythonProject1\Models\best_logreg_model.pkl'
model_label_encoders_path = r'E:\Giza Systems\Model Evaluation\Ali_Badawy\pythonProject1\Models\label_encoders.pkl'
model_scaler_path = r'E:\Giza Systems\Model Evaluation\Ali_Badawy\pythonProject1\Models\scaler.pkl'

logreg_model = joblib.load(model_logreg_path)
label_encoders = joblib.load(model_label_encoders_path)
scaler = joblib.load(model_scaler_path)

# Define features
features = ['Num_Children', 'Gender', 'Income', 'Own_Car', 'Own_Housing']
categorical_features = ['Gender', 'Own_Car', 'Own_Housing']
numerical_features = ['Income', 'Num_Children']
required_features = features

# Path to the test data file and API URL
test_data_path = r"E:\Giza Systems\Model Evaluation\credit_card_test.csv"
API_URL = "http://127.0.0.1:8000/api/predict/"

# Load and preprocess test data from the file for both prediction and SHAP
def load_test_data():
    # Load the test data as a DataFrame
    data = pd.read_csv(test_data_path)

    # Prepare data for model prediction
    input_data = data[["Num_Children", "Gender", "Income", "Own_Car", "Own_Housing"]]
    target_data = data["Credit_Card_Issuing"]

    return input_data.to_dict(orient="list"), target_data, data

# Preprocess data for SHAP analysis
def preprocess_for_shap(df):
    # Apply the same preprocessing steps as in Django's views.py
    # Encode categorical features using label encoders
    for col in categorical_features:
        le = label_encoders[col]
        try:
            df[col] = le.transform(df[col])
        except ValueError as ve:
            print(f"Warning: {ve} - Check for unseen categories in the test data.")

    # Scale numerical features
    df[numerical_features] = scaler.transform(df[numerical_features])

    # Ensure correct column order
    df = df[features]

    return df

# Function to send data to the prediction server and get predictions
def send_to_prediction_api(data):
    with httpx.Client() as client:
        try:
            response = client.post(API_URL, json=data)
            response.raise_for_status()
            return response.json().get("predictions", [])
        except httpx.RequestError as e:
            print(f"Request failed: {e}")
            return []

# SHAP analysis function
def shap_analysis(df):
    explainer = shap.Explainer(logreg_model, df)
    shap_values = explainer(df)

    # Generate SHAP summary plot
    shap.summary_plot(shap_values, df, show=False)
    plt.show()

# Bin Income into categories for fairness analysis
def categorize_income(income):
    if income < 30000:
        return "Low"
    elif income < 70000:
        return "Medium"
    else:
        return "High"

# Calculate subgroup accuracy for a specific feature
def calculate_subgroup_accuracy(df, predictions, actuals, feature_name):
    subgroup_accuracies = {}
    actuals_binary = [1 if actual == "Approved" else 0 for actual in actuals]
    for value in df[feature_name].unique():
        subgroup = df[df[feature_name] == value]
        indices = subgroup.index
        accuracy = accuracy_score([actuals_binary[i] for i in indices], [predictions[i] for i in indices])
        subgroup_accuracies[value] = accuracy
    return subgroup_accuracies

# Calculate fairness metrics: Statistical Parity, Equal Opportunity, Equalized Odds
def calculate_fairness_metrics(df, predictions, actuals, sensitive_attribute):
    fairness_metrics = {}
    actuals_binary = [1 if actual == "Approved" else 0 for actual in actuals]
    group_rates = df.groupby(sensitive_attribute).apply(
        lambda x: sum(predictions[i] for i in x.index) / len(x)
    )
    fairness_metrics["Statistical Parity"] = group_rates.to_dict()
    tpr_by_group = df[df[sensitive_attribute].notna()].groupby(sensitive_attribute).apply(
        lambda x: recall_score([actuals_binary[i] for i in x.index], [predictions[i] for i in x.index])
    )
    fairness_metrics["Equal Opportunity (TPR)"] = tpr_by_group.to_dict()
    fpr_by_group = df[df[sensitive_attribute].notna()].groupby(sensitive_attribute).apply(
        lambda x: sum(1 for i in x.index if actuals_binary[i] == 0 and predictions[i] == 1) / sum(
            1 for i in x.index if actuals_binary[i] == 0)
    )
    fairness_metrics["Equalized Odds (TPR)"] = tpr_by_group.to_dict()
    fairness_metrics["Equalized Odds (FPR)"] = fpr_by_group.to_dict()
    return fairness_metrics

# Calculate model evaluation, fairness metrics, variance of numerical features, and subgroup accuracy
def calculate_metrics(predictions, actuals, df):
    if not predictions:
        print("No predictions received from the server.")
        return {"error": "No predictions received from the server."}
    actuals_binary = [1 if actual == "Approved" else 0 for actual in actuals]
    performance = {
        "accuracy": float(accuracy_score(actuals_binary, predictions)),
        "precision": float(precision_score(actuals_binary, predictions)),
        "recall": float(recall_score(actuals_binary, predictions)),
        "f1_score": float(f1_score(actuals_binary, predictions))
    }
    variance_results = df[numerical_features].var().to_dict()
    sensitive_attributes = ["Gender", "Own_Car", "Own_Housing", "Income_Category"]
    fairness_results = {}
    for attribute in sensitive_attributes:
        fairness_results[attribute] = calculate_fairness_metrics(df, predictions, actuals, attribute)
    subgroup_accuracy_results = {}
    for attribute in sensitive_attributes:
        subgroup_accuracy_results[attribute] = calculate_subgroup_accuracy(df, predictions, actuals, attribute)
    return {
        "performance": performance,
        "variance": variance_results,
        "fairness": fairness_results,
        "subgroup_accuracy": subgroup_accuracy_results
    }

# Main function to perform the evaluation
def main():
    input_data_dict, actual_labels, df = load_test_data()

    # Add Income categories to the dataframe for fairness analysis
    df["Income_Category"] = df["Income"].apply(categorize_income)

    predictions = send_to_prediction_api(input_data_dict)
    preprocessed_df = preprocess_for_shap(df.copy())  # Preprocess for SHAP
    shap_analysis(preprocessed_df)  # Perform SHAP analysis
    results = calculate_metrics(predictions, actual_labels, df)
    print("Evaluation Results:")
    for key, value in results.items():
        if key == "fairness":
            print(f"\nFairness Metrics:")
            for attribute, metrics in value.items():
                print(f"\nAttribute: {attribute}")
                for metric_name, metric_value in metrics.items():
                    print(f"  {metric_name}: {metric_value}")
        elif key == "variance":
            print("\nVariance of Numerical Features:")
            for feature, variance in value.items():
                print(f"  {feature}: {variance}")
        elif key == "subgroup_accuracy":
            print("\nSubgroup Accuracy by Sensitive Attribute:")
            for attribute, subgroup_accuracies in value.items():
                print(f"\nAttribute: {attribute}")
                for subgroup, accuracy in subgroup_accuracies.items():
                    print(f"  {subgroup}: {accuracy}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
