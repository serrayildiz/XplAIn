import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch.optim as optim
import joblib

# Define CNN Model
class CNNModel(nn.Module):
    def __init__(self, input_size, output_size=2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TrainEvaluateCNN:
    def __init__(self, model, train_loader, val_loader, epochs=20, lr=0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.best_accuracy = 0

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            val_loss, val_accuracy = self.evaluate()
            print(f'Epoch {epoch + 1}: Loss: {running_loss / len(self.train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f'Saving new best model with accuracy: {val_accuracy:.2f}%')

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return total_loss / len(self.val_loader), 100 * correct / total

    def test(self, test_loader):
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy

def preprocess_data(df, categorical_columns, numerical_columns):
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le  # Save encoder

    # Save the label encoders
    joblib.dump(label_encoders, 'label_encoders.pkl')

    # Normalize numerical columns
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')

    # Convert boolean columns to integer (0 and 1)
    boolean_columns = df.select_dtypes(include=['bool']).columns
    df[boolean_columns] = df[boolean_columns].astype(int)

    # Map categorical labels to integers
    class_mapping = {'bad': 0, 'good': 1}
    df['class'] = df['class'].map(class_mapping)

    return df

def create_dataloaders(df):
    X = df.drop('class', axis=1).values
    y = df['class'].values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    df = pd.read_csv('credit_scoring.csv')

    categorical_columns = ['checking_status', 'credit_history', 'purpose', 'savings_status', 
                           'employment', 'personal_status', 'other_parties', 'property_magnitude', 
                           'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker']
    numerical_columns = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 
                         'existing_credits', 'num_dependents']

    df = preprocess_data(df, categorical_columns, numerical_columns)
    train_loader, test_loader = create_dataloaders(df)

    model = CNNModel(input_size=20)
    trainer = TrainEvaluateCNN(model, train_loader, test_loader, epochs=100, lr=0.005)
    trainer.train()
    trainer.test(test_loader)
