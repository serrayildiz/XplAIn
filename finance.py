import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Load your dataset
df = pd.read_csv('loan_data.csv')

# Drop the Loan_ID column and add a generic ID column
df.drop('Loan_ID', axis=1, inplace=True)
df['ID'] = range(1, len(df) + 1)

# Convert categorical variables to numerical format using one-hot encoding
categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Convert True/False to 1/0
boolean_columns = ['Married_Yes', 'Education_Not Graduate', 'Self_Employed_Yes', 'Property_Area_Semiurban', 'Property_Area_Urban', 'Loan_Status_Y']
df[boolean_columns] = df[boolean_columns].astype(int)

# Handle the 'Dependents' column, including missing values
df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0]).astype(int)

# Handle missing values in numerical columns
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

# Split the dataset into features (X) and target (y)
X = df.drop('Loan_Status_Y', axis=1)
y = df['Loan_Status_Y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Explain a prediction using LIME
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=['Rejected', 'Approved'],
    mode='classification'
)

# Explain the first instance in the test set
exp = explainer.explain_instance(X_test.iloc[0], model.predict_proba)

# Print the coefficients for each feature
coefficients = exp.as_list()
for feature, value in coefficients:
    print(f"{feature}: {value}")
