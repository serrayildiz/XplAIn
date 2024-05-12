import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import shap
import joblib
from credit_cnn import CNNModel

def predict_and_explain(model, test_data, feature_names, original_data):
    model.eval()
    with torch.no_grad():
        # Predict using your model
        outputs = model(test_data)
        _, predicted = torch.max(outputs, 1)

        # Initialize IntegratedGradients
        ig = IntegratedGradients(model)

        # Choose a target class
        target_class = predicted.item()
        
        # Get attributions
        attributions, delta = ig.attribute(test_data, target=target_class, return_convergence_delta=True)

    attributions = attributions.detach().numpy()

    # Prepare SHAP Gradient Explainer
    explainer = shap.GradientExplainer(model, torch.tensor(original_data.values.astype(np.float32)))
    shap_values = explainer.shap_values(test_data)

    # Extract SHAP values for the target class
    shap_values_for_target = shap_values[target_class][0]

    # Prepare DataFrame for output
    data = {
        'Name': feature_names,
        'Contribution': attributions.squeeze().tolist(),
        'Causal Effects': shap_values_for_target.tolist()
    }
    df = pd.DataFrame(data)

    # Map the predicted class to "Bad" or "Good"
    class_labels = {0: "Bad", 1: "Good"}
    predicted_label = class_labels.get(predicted.item(), "Unknown")

    return df, delta, predicted_label


def preprocess_data(new_df, categorical_columns, numerical_columns):
    # Load saved encoders and scaler
    label_encoders = joblib.load('label_encoders.pkl')
    scaler = joblib.load('scaler.pkl')

    # Fill missing values and encode categorical columns
    new_df[numerical_columns] = new_df[numerical_columns].fillna(new_df[numerical_columns].mean())
    for column in categorical_columns:
        le = label_encoders[column]
        new_df[column] = le.transform(new_df[column])

    # Normalize numerical columns using the loaded scaler
    new_df[numerical_columns] = scaler.transform(new_df[numerical_columns])

    # Convert boolean columns to integer (0 and 1)
    boolean_columns = new_df.select_dtypes(include=['bool']).columns
    new_df[boolean_columns] = new_df[boolean_columns].astype(int)

    return new_df

if __name__ == "__main__":
    new_df = pd.read_csv('test.csv')

    categorical_columns = ['checking_status', 'credit_history', 'purpose', 'savings_status', 
                           'employment', 'personal_status', 'other_parties', 'property_magnitude', 
                           'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker']
    numerical_columns = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 
                         'existing_credits', 'num_dependents']

    # Preprocess the new data
    new_df = preprocess_data(new_df, categorical_columns, numerical_columns)
    feature_names = new_df.columns.tolist()

    # Convert the preprocessed dataframe to tensors
    X_new = new_df.values.astype(np.float32)
    test_data = torch.tensor(X_new)  # Convert to tensor

    # Prompt user to choose a row index
    row_index = int(input("Enter the index of the row you want to explain: "))

    # Select the chosen row
    selected_row = test_data[row_index].unsqueeze(0)  # Add batch dimension

    model = CNNModel(input_size=X_new.shape[1])
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Example call to the function for explaining prediction on the selected row
    explanation_df, convergence_delta, predicted_class = predict_and_explain(model, selected_row, feature_names, new_df)

    # Print or use the explanation dataframe as needed
    print(explanation_df)
    print("Predicted class:", predicted_class)

