# import os
# import sys
# import django
# import httpx
# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import numpy as np
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
#
# # Set up the Django environment
# # Ensure this path points to the Django project root directory
# django_project_path = r'E:\Giza Systems\Model Evaluation\Yara_Mahfouz\pythonProject'
# sys.path.append(django_project_path)
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'credit_card_approval_prediction.settings')
#
# # Attempt to set up Django if needed (comment out if Django setup is unnecessary)
# try:
#     django.setup()
# except Exception as e:
#     print(f"Django setup failed: {e}")
#
# # Path to the test data file
# test_data_path = r"E:\Giza Systems\Model Evaluation\credit_card_test.csv"  # Update this path to your file
# API_URL = "http://127.0.0.1:8000/my_app/predict/"
#
#
# # Load and preprocess test data from the file
# def load_test_data():
#     # Load the test data as a DataFrame
#     data = pd.read_csv(test_data_path)
#
#     # Prepare data for model prediction
#     input_data = data[["Num_Children", "Gender", "Income", "Own_Car", "Own_Housing"]]
#     target_data = data["Credit_Card_Issuing"]
#
#     return input_data.to_dict(orient="list"), target_data, data
#
#
# # Function to send data to the prediction server and get predictions
# def send_to_prediction_api(data):
#     with httpx.Client() as client:
#         try:
#             response = client.post(API_URL, json=data)
#             response.raise_for_status()  # Raise an error for any 4xx/5xx status codes
#             return response.json().get("predictions", [])
#         except httpx.RequestError as e:
#             print(f"Request failed: {e}")
#             return []
#
#
# # Bin Income into categories for fairness analysis
# def categorize_income(income):
#     if income < 30000:
#         return "Low"
#     elif income < 70000:
#         return "Medium"
#     else:
#         return "High"
#
#
# # Calculate subgroup accuracy for a specific feature
# def calculate_subgroup_accuracy(df, predictions, actuals, feature_name):
#     subgroup_accuracies = {}
#
#     # Convert actuals and predictions to binary format for accuracy calculation
#     actuals_binary = [1 if actual == "Approved" else 0 for actual in actuals]
#     # Calculate accuracy for each unique value in the feature
#     for value in df[feature_name].unique():
#         subgroup = df[df[feature_name] == value]
#
#         # Get the indices of the current subgroup
#         indices = subgroup.index
#
#         # Calculate accuracy for this subgroup
#         accuracy = accuracy_score([actuals_binary[i] for i in indices], [predictions[i] for i in indices])
#         subgroup_accuracies[value] = accuracy
#
#     return subgroup_accuracies
#
#
# # Calculate fairness metrics: Statistical Parity, Equal Opportunity, Equalized Odds
# def calculate_fairness_metrics(df, predictions, actuals, sensitive_attribute):
#     fairness_metrics = {}
#
#     # Convert actuals to binary
#     actuals_binary = [1 if actual == "Approved" else 0 for actual in actuals]
#
#     # Statistical Parity: Positive prediction rates by group
#     group_rates = df.groupby(sensitive_attribute).apply(
#         lambda x: sum(predictions[i] for i in x.index) / len(x)
#     )
#     fairness_metrics["Statistical Parity"] = group_rates.to_dict()
#
#     # Equal Opportunity: True Positive Rates (TPR) by group
#     tpr_by_group = df[df[sensitive_attribute].notna()].groupby(sensitive_attribute).apply(
#         lambda x: recall_score([actuals_binary[i] for i in x.index], [predictions[i] for i in x.index])
#     )
#     fairness_metrics["Equal Opportunity (TPR)"] = tpr_by_group.to_dict()
#
#     # Equalized Odds: TPR and FPR by group
#     fpr_by_group = df[df[sensitive_attribute].notna()].groupby(sensitive_attribute).apply(
#         lambda x: sum(1 for i in x.index if actuals_binary[i] == 0 and predictions[i] == 1) / sum(
#             1 for i in x.index if actuals_binary[i] == 0)
#     )
#     fairness_metrics["Equalized Odds (TPR)"] = tpr_by_group.to_dict()
#     fairness_metrics["Equalized Odds (FPR)"] = fpr_by_group.to_dict()
#
#     return fairness_metrics
#
#
# # Calculate model evaluation, fairness metrics, variance of numerical features, and subgroup accuracy
# def calculate_metrics(predictions, actuals, df):
#     if not predictions:
#         print("No predictions received from the server.")
#         return {"error": "No predictions received from the server."}
#
#     # Convert actuals to binary format
#     actuals_binary = [1 if actual == "Approved" else 0 for actual in actuals]
#
#     # Calculate performance metrics
#     performance = {
#         "accuracy": float(accuracy_score(actuals_binary, predictions)),
#         "precision": float(precision_score(actuals_binary, predictions)),
#         "recall": float(recall_score(actuals_binary, predictions)),
#         "f1_score": float(f1_score(actuals_binary, predictions))
#     }
#
#     # Calculate variance for numerical features
#     numerical_features = ["Num_Children", "Income"]
#     variance_results = df[numerical_features].var().to_dict()
#
#     # Calculate fairness metrics for each sensitive attribute
#     sensitive_attributes = ["Gender", "Own_Car", "Own_Housing", "Income_Category"]
#     fairness_results = {}
#     for attribute in sensitive_attributes:
#         fairness_results[attribute] = calculate_fairness_metrics(df, predictions, actuals, attribute)
#
#     # Calculate subgroup accuracy for each sensitive attribute
#     subgroup_accuracy_results = {}
#     for attribute in sensitive_attributes:
#         subgroup_accuracy_results[attribute] = calculate_subgroup_accuracy(df, predictions, actuals, attribute)
#
#     return {
#         "performance": performance,
#         "variance": variance_results,
#         "fairness": fairness_results,
#         "subgroup_accuracy": subgroup_accuracy_results
#     }
#
#
# # Main function to perform the evaluation
# def main():
#     # Load test data
#     input_data, actual_labels, df = load_test_data()
#
#     # Add Income categories to the dataframe for fairness analysis
#     df["Income_Category"] = df["Income"].apply(categorize_income)
#
#     # Send test data to the prediction server and get predictions
#     predictions = send_to_prediction_api(input_data)
#
#     # Calculate evaluation metrics
#     results = calculate_metrics(predictions, actual_labels, df)
#
#     # Print the evaluation results
#     print("Evaluation Results:")
#     for key, value in results.items():
#         if key == "fairness":
#             print(f"\nFairness Metrics:")
#             for attribute, metrics in value.items():
#                 print(f"\nAttribute: {attribute}")
#                 for metric_name, metric_value in metrics.items():
#                     print(f"  {metric_name}: {metric_value}")
#         elif key == "variance":
#             print("\nVariance of Numerical Features:")
#             for feature, variance in value.items():
#                 print(f"  {feature}: {variance}")
#         elif key == "subgroup_accuracy":
#             print("\nSubgroup Accuracy by Sensitive Attribute:")
#             for attribute, subgroup_accuracies in value.items():
#                 print(f"\nAttribute: {attribute}")
#                 for subgroup, accuracy in subgroup_accuracies.items():
#                     print(f"  {subgroup}: {accuracy}")
#         else:
#             print(f"{key}: {value}")
#
#
# # Run the main function
# if __name__ == "__main__":
#     main()
#

import os
import sys
import django
import httpx
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up the Django environment
django_project_path = r'E:\Giza Systems\Model Evaluation\Yara_Mahfouz\pythonProject'
sys.path.append(django_project_path)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'credit_card_approval_prediction.settings')

try:
    django.setup()
except Exception as e:
    print(f"Django setup failed: {e}")

# Paths to the model, scaler, and test data file
model_path = "my_app/model/model.pkl"
scaler_path = "my_app/model/scaler.pkl"
test_data_path = r"E:\Giza Systems\Model Evaluation\credit_card_test.csv"
API_URL = "http://127.0.0.1:8000/my_app/predict/"

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load and preprocess test data from the file
def load_test_data():
    data = pd.read_csv(test_data_path)
    input_data = data[["Num_Children", "Gender", "Income", "Own_Car", "Own_Housing"]]
    target_data = data["Credit_Card_Issuing"]
    return input_data.to_dict(orient="list"), target_data, data

# Preprocess data for SHAP analysis based on server preprocessing logic
def preprocess_for_shap(data):
    df = pd.DataFrame(data)

    # Map categorical fields
    gender_mapping = {'Male': 1, 'Female': 0}
    car_mapping = {'Yes': 1, 'No': 0}
    housing_mapping = {'Yes': 1, 'No': 0}
    df['Gender'] = df['Gender'].map(gender_mapping)
    df['Own_Car'] = df['Own_Car'].map(car_mapping)
    df['Own_Housing'] = df['Own_Housing'].map(housing_mapping)

    # Normalize the Income column using the loaded scaler
    df['Income'] = scaler.transform(df[['Income']])

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


def shap_analysis(data):
    # Preprocess data for SHAP analysis
    preprocessed_data = preprocess_for_shap(data)
    print("Preprocessed data columns:", preprocessed_data.columns)
    print("Preprocessed data shape:", preprocessed_data.shape)

    # Initialize the SHAP Explainer with predict_proba for probability explanations
    explainer = shap.Explainer(model.predict_proba, preprocessed_data)
    shap_values = explainer(preprocessed_data)

    # Check and print the shape of shap_values
    print("SHAP values shape:", getattr(shap_values, 'shape', 'Not available'))

    # Selecting SHAP values for the positive class (class 1)
    shap_values_positive_class = shap_values[..., 1]  # Select SHAP values for class 1

    # Double-check that the SHAP values align with the preprocessed data
    if shap_values_positive_class.shape != preprocessed_data.shape:
        print(f"Mismatch in shapes! SHAP values shape: {shap_values_positive_class.shape}, "
              f"Preprocessed data shape: {preprocessed_data.shape}")
        return  # Stop further execution if there's a shape mismatch

    # Plot summary with SHAP for the positive class
    shap.summary_plot(shap_values_positive_class, preprocessed_data, show=False)
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

# Calculate fairness metrics
def calculate_fairness_metrics(df, predictions, actuals, sensitive_attribute):
    fairness_metrics = {}
    actuals_binary = [1 if actual == "Approved" else 0 for actual in actuals]
    group_rates = df.groupby(sensitive_attribute).apply(lambda x: sum(predictions[i] for i in x.index) / len(x))
    fairness_metrics["Statistical Parity"] = group_rates.to_dict()
    tpr_by_group = df[df[sensitive_attribute].notna()].groupby(sensitive_attribute).apply(
        lambda x: recall_score([actuals_binary[i] for i in x.index], [predictions[i] for i in x.index])
    )
    fairness_metrics["Equal Opportunity (TPR)"] = tpr_by_group.to_dict()
    fpr_by_group = df[df[sensitive_attribute].notna()].groupby(sensitive_attribute).apply(
        lambda x: sum(1 for i in x.index if actuals_binary[i] == 0 and predictions[i] == 1) /
                  sum(1 for i in x.index if actuals_binary[i] == 0)
    )
    fairness_metrics["Equalized Odds (TPR)"] = tpr_by_group.to_dict()
    fairness_metrics["Equalized Odds (FPR)"] = fpr_by_group.to_dict()
    return fairness_metrics

# Calculate metrics
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

# Main function
def main():
    input_data, actual_labels, df = load_test_data()
    df["Income_Category"] = df["Income"].apply(categorize_income)

    predictions = send_to_prediction_api(input_data)
    if predictions:
        print("Predictions received from the server:", predictions)
        shap_analysis(input_data)  # SHAP analysis
        results = calculate_metrics(predictions, actual_labels, df)
        print("Evaluation Results:")
        for key, value in results.items():
            if key == "fairness":
                print("\nFairness Metrics:")
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
