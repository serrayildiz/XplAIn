# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime import lime_tabular
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name='target')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Explain a prediction using LIME
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=cancer.target_names,
    mode='classification'
)

# Explain the first instance in the test set
exp = explainer.explain_instance(X_test.iloc[0], model.predict_proba)

# Print the coefficients for each feature
coefficients = exp.as_list()
for feature, value in coefficients:
    print(f"{feature}: {value}")
