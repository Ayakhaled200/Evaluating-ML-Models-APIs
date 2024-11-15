import httpx
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Paths to the test data, model, and API URL
test_data_path = r"E:\Giza Systems\Model Evaluation\credit_card_test.csv"  # Update this path to your file
API_URL = "http://127.0.0.1:5000/predict"  # Replace with the actual prediction API URL
model_path = "xgb_model.pkl"  # Model path for SHAP

# Load the model locally for SHAP analysis
model = joblib.load(model_path)


# Load and preprocess test data from the file
def load_test_data():
    data = pd.read_csv(test_data_path)
    input_data = data[["Num_Children", "Gender", "Income", "Own_Car", "Own_Housing"]]
    target_data = data["Credit_Card_Issuing"]
    return input_data.to_dict(orient="list"), target_data, data


def preprocess_for_shap(data):
    df = pd.DataFrame({
        'Num_Children': data['Num_Children'],
        'Gender': data['Gender'],
        'Income': data['Income'],
        'Own_Car': data['Own_Car'],
        'Own_Housing': data['Own_Housing']
    })

    # Apply encoding and normalization steps
    df['Gender'] = df['Gender'].replace({'Male': True, 'Female': False}).astype(bool)
    df['Own_Car'] = df['Own_Car'].replace({'Yes': True, 'No': False}).astype(bool)
    df['Own_Housing'] = df['Own_Housing'].replace({'Yes': True, 'No': False}).astype(bool)

    # Ensure the DataFrame matches expected columns
    expected_columns = ['Num_Children', 'Gender', 'Income', 'Own_Car', 'Own_Housing']
    df = df[expected_columns].astype(float)

    return df

# Function to send data to the prediction server and get predictions
def send_to_prediction_api(data):
    with httpx.Client() as client:
        try:
            response = client.post(API_URL, json=data)
            response.raise_for_status()  # Raise an error for any 4xx/5xx status codes
            return response.json().get("predictions", [])
        except httpx.RequestError as e:
            print(f"Request failed: {e}")
            return []


# SHAP analysis function
def shap_analysis(data):
    explainer = shap.Explainer(model, data)  # SHAP automatically chooses the right explainer
    shap_values = explainer(data)
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

    # Performance metrics
    performance = {
        "accuracy": float(accuracy_score(actuals_binary, predictions)),
        "precision": float(precision_score(actuals_binary, predictions)),
        "recall": float(recall_score(actuals_binary, predictions)),
        "f1_score": float(f1_score(actuals_binary, predictions))
    }

    # Variance for numerical features
    numerical_features = ["Num_Children", "Income"]
    variance_results = df[numerical_features].var().to_dict()

    # Fairness metrics for each sensitive attribute
    sensitive_attributes = ["Gender", "Own_Car", "Own_Housing", "Income_Category"]
    fairness_results = {}
    for attribute in sensitive_attributes:
        fairness_results[attribute] = calculate_fairness_metrics(df, predictions, actuals, attribute)

    # Subgroup accuracy for each sensitive attribute
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
    # Load test data
    input_data, actual_labels, df = load_test_data()

    # Add Income categories to the dataframe for fairness analysis
    df["Income_Category"] = df["Income"].apply(categorize_income)

    # Send test data to the prediction server and get predictions
    predictions = send_to_prediction_api(input_data)

    # Calculate evaluation metrics
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

    # Perform SHAP analysis
    shap_data = preprocess_for_shap(df[["Num_Children", "Gender", "Income", "Own_Car", "Own_Housing"]])
    shap_analysis(shap_data)


# Run the main function
if __name__ == "__main__":
    main()
