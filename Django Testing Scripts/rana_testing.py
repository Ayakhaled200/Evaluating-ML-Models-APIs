import os
import sys
import django
import httpx
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up the Django environment
django_project_path = r'E:\Giza Systems\Model Evaluation\Rana Hosny\pythonProject\Predictions_app'
sys.path.append(django_project_path)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Predictions_app.settings')

try:
    django.setup()
except Exception as e:
    print(f"Django setup failed: {e}")

# Paths
test_data_path = r"E:\Giza Systems\Model Evaluation\credit_card_test.csv"
API_URL = "http://127.0.0.1:8000/api/predict/"
model_path = r'E:\Giza Systems\Model Evaluation\Rana Hosny\pythonProject\Predictions_app\predictions_api\LogisticRegression_best_model.pkl'

# Load and preprocess test data
def load_test_data():
    data = pd.read_csv(test_data_path)
    input_data = data[["Income", "Own_Car", "Own_Housing", "Gender", "Num_Children"]]
    target_data = data["Credit_Card_Issuing"]
    return input_data.to_dict(orient="list"), target_data, data

def preprocess_for_shap(data):
    df = pd.DataFrame(data)

    # Map categorical fields
    df['Own_Car'] = df['Own_Car'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Own_Housing'] = df['Own_Housing'].apply(lambda x: 1 if x == 'Yes' else 0)

    # One-hot encode Gender (align with model training)
    df = pd.get_dummies(df, columns=['Gender'], drop_first=False)

    # Ensure only the features expected by the model
    feature_names = ['Income', 'Own_Car', 'Own_Housing', 'Gender_Female', 'Gender_Male', 'Num_Children']
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0  # Add missing columns with default value of 0

    df = df[feature_names]  # Ensure column order matches the model
    return df.astype(float)


# Function to send data to the prediction server
def send_to_prediction_api(data):
    with httpx.Client() as client:
        try:
            response = client.post(API_URL, json=data)
            response.raise_for_status()
            return response.json().get("prediction", [])
        except httpx.RequestError as e:
            print(f"Request failed: {e}")
            return []


def shap_analysis(data):
    # Load the dictionary from the model file
    model_dict = joblib.load(model_path)

    # Extract the actual model
    actual_model = model_dict.get('model')

    if not actual_model or not hasattr(actual_model, 'predict'):
        raise AttributeError("The extracted model does not have a 'predict' method.")

    # Initialize SHAP Explainer
    explainer = shap.Explainer(actual_model.predict, data)
    shap_values = explainer(data)

    # Plot SHAP summary
    shap.summary_plot(shap_values, data, show=False)
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
        "accuracy": accuracy_score(actuals_binary, predictions),
        "precision": precision_score(actuals_binary, predictions),
        "recall": recall_score(actuals_binary, predictions),
        "f1_score": f1_score(actuals_binary, predictions)
    }
    numerical_features = ["Num_Children", "Income"]
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


def main():
    input_data, actual_labels, df = load_test_data()
    df["Income_Category"] = df["Income"].apply(categorize_income)

    # Preprocess for SHAP
    shap_data = preprocess_for_shap(df)

    # Send input data to the server and get predictions
    predictions = send_to_prediction_api(input_data)

    if predictions:
        print("Predictions received from the server:", predictions)
        shap_analysis(shap_data)
        results = calculate_metrics(predictions, actual_labels, df)

        # Print evaluation results
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
    else:
        print("No predictions received from the server.")


if __name__ == "__main__":
    main()