import httpx
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

test_data_path = r"E:\Giza Systems\Model Evaluation\credit_card_test.csv"
API_URL = "http://127.0.0.1:5000/predict"

# Load scaler for preprocessing (same scaler used in the server)
scaler = joblib.load('scaler.pkl')

MODEL_FEATURE_ORDER = [
    'Num_Children', 'Income', 'Income_per_Child', 'Total_Owned_Assets',
    'Income_Interaction', 'Income_Stability_Score', 'Gender', 'Own_Housing',
    'Own_Car', 'Financial_Stability', 'Large_Family', 'Gender_Family_Interaction'
]
binary_list = ['Gender', 'Own_Housing', 'Own_Car', 'Financial_Stability', 'Large_Family', 'Gender_Family_Interaction']


# Load and preprocess test data
def load_test_data():
    data = pd.read_csv(test_data_path)
    input_data = data[["Num_Children", "Gender", "Income", "Own_Car", "Own_Housing"]]
    target_data = data["Credit_Card_Issuing"]
    return input_data.to_dict(orient="list"), target_data, data


# Preprocessing function for SHAP analysis
def preprocess_for_shap(data):
    # Create a deep copy of the data to avoid SettingWithCopyWarning
    data = data.copy()

    # Map categorical variables to numerical values
    data.loc[:, 'Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data.loc[:, 'Own_Car'] = data['Own_Car'].map({'Yes': 1, 'No': 0})
    data.loc[:, 'Own_Housing'] = data['Own_Housing'].map({'Yes': 1, 'No': 0})

    # Feature engineering steps
    data.loc[:, 'Income_per_Child'] = data['Income'] / (data['Num_Children'] + 1)
    data.loc[:, 'Financial_Stability'] = ((data['Own_Car'] == 1) & (data['Own_Housing'] == 1)).astype(int)
    data.loc[:, 'Large_Family'] = (data['Num_Children'] > 3).astype(int)
    data.loc[:, 'Total_Owned_Assets'] = data['Own_Car'] + data['Own_Housing']
    data.loc[:, 'Gender_Family_Interaction'] = data['Gender'] * data['Num_Children']
    data.loc[:, 'Income_Interaction'] = data['Income'] * data['Num_Children']
    data.loc[:, 'Income_Stability_Score'] = data['Income'] * (data['Own_Housing'] + data['Own_Car'])

    # Separate binary and continuous columns for scaling
    data_to_scale = data.drop(columns=binary_list)
    data_scaled_values = scaler.transform(data_to_scale)
    data_scaled = pd.DataFrame(data_scaled_values, columns=data_to_scale.columns)

    # Add binary features back, ensuring they're integers
    for col in binary_list:
        data_scaled[col] = data[col].astype(int).values  # Convert to int

    # Ensure correct column order and data types
    data = data_scaled[MODEL_FEATURE_ORDER].astype(float)  # Convert all columns to float for compatibility
    return data


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

    # Convert actuals and predictions to binary format for accuracy calculation
    actuals_binary = [1 if actual == "Approved" else 0 for actual in actuals]
    # Calculate accuracy for each unique value in the feature
    for value in df[feature_name].unique():
        subgroup = df[df[feature_name] == value]

        # Get the indices of the current subgroup
        indices = subgroup.index

        # Calculate accuracy for this subgroup
        accuracy = accuracy_score([actuals_binary[i] for i in indices], [predictions[i] for i in indices])
        subgroup_accuracies[value] = accuracy

    return subgroup_accuracies


# Calculate fairness metrics: Statistical Parity, Equal Opportunity, Equalized Odds
def calculate_fairness_metrics(df, predictions, actuals, sensitive_attribute):
    fairness_metrics = {}

    # Convert actuals to binary
    actuals_binary = [1 if actual == "Approved" else 0 for actual in actuals]

    # Statistical Parity: Positive prediction rates by group
    group_rates = df.groupby(sensitive_attribute).apply(
        lambda x: sum(predictions[i] for i in x.index) / len(x)
    )
    fairness_metrics["Statistical Parity"] = group_rates.to_dict()

    # Equal Opportunity: True Positive Rates (TPR) by group
    tpr_by_group = df[df[sensitive_attribute].notna()].groupby(sensitive_attribute).apply(
        lambda x: recall_score([actuals_binary[i] for i in x.index], [predictions[i] for i in x.index])
    )
    fairness_metrics["Equal Opportunity (TPR)"] = tpr_by_group.to_dict()

    # Equalized Odds: TPR and FPR by group
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

    # Convert actuals to binary format
    actuals_binary = [1 if actual == "Approved" else 0 for actual in actuals]

    # Calculate performance metrics
    performance = {
        "accuracy": float(accuracy_score(actuals_binary, predictions)),
        "precision": float(precision_score(actuals_binary, predictions)),
        "recall": float(recall_score(actuals_binary, predictions)),
        "f1_score": float(f1_score(actuals_binary, predictions))
    }

    # Calculate variance for numerical features
    numerical_features = ["Num_Children", "Income"]
    variance_results = df[numerical_features].var().to_dict()

    # Calculate fairness metrics for each sensitive attribute
    sensitive_attributes = ["Gender", "Own_Car", "Own_Housing", "Income_Category"]
    fairness_results = {}
    for attribute in sensitive_attributes:
        fairness_results[attribute] = calculate_fairness_metrics(df, predictions, actuals, attribute)

    # Calculate subgroup accuracy for each sensitive attribute
    subgroup_accuracy_results = {}
    for attribute in sensitive_attributes:
        subgroup_accuracy_results[attribute] = calculate_subgroup_accuracy(df, predictions, actuals, attribute)

    return {
        "performance": performance,
        "variance": variance_results,
        "fairness": fairness_results,
        "subgroup_accuracy": subgroup_accuracy_results
    }

# Main function to perform the evaluation and SHAP analysis
def main():
    # Load test data
    input_data, actual_labels, df = load_test_data()

    df["Income_Category"] = df["Income"].apply(categorize_income)

    # Send test data to the prediction server and get predictions
    predictions = send_to_prediction_api(input_data)

    # Perform SHAP analysis
    shap_data = preprocess_for_shap(df[["Num_Children", "Gender", "Income", "Own_Car", "Own_Housing"]])

    # Initialize SHAP explainer with the trained model
    model = xgb.XGBClassifier()
    model.load_model('credit_card_model.json')
    explainer = shap.Explainer(model)

    # Compute SHAP values
    shap_values = explainer(shap_data)
    shap.summary_plot(shap_values, shap_data, show=False)  # Generates a SHAP summary plot
    plt.show()

    results = calculate_metrics(predictions, actual_labels, df)

    # Print the evaluation results
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
