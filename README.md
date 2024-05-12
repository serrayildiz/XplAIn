# Credit Scoring CNN Project with XAI

This project implements a Convolutional Neural Network (CNN) for credit scoring predictions. The model is built using PyTorch and explains predictions using Captum and SHAP to enhance transparency and interpretability in machine learning.

## Project Structure

- **credit_cnn.py**: Training script for the CNN model.
- **explain_cnn.py**: Script to explain the predictions using Captum and SHAP.
- **credit_scoring.csv**: Dataset for training the model.
- **test.csv**: Dataset for making predictions and explanations.
- **best_model.pth**: Saved model with the best validation accuracy.
- **label_encoders.pkl**: Saved LabelEncoder objects for categorical features.
- **scaler.pkl**: Saved StandardScaler object for numerical features.

## Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/mehmetuzunyayla/xplain.git
    cd xplain
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the Data**:
    - Ensure `credit_scoring.csv` is in the project directory for training.
    - Ensure `test.csv` is in the project directory for testing and explanation.

4. **Training the Model**:
    ```bash
    python credit_cnn.py
    ```

5. **Explaining Predictions**:
    ```bash
    python explain_cnn.py
    ```

## Model Architecture

The CNN model consists of:
- Three convolutional layers with Batch Normalization and LeakyReLU activation.
- One dropout layer to prevent overfitting.
- Two fully connected layers.

## Explanation Techniques

- **Captum IntegratedGradients**: To calculate feature attributions.
- **SHAP GradientExplainer**: To provide additional insights into feature contributions.

## Example Usage

- **Training Output**:
    ```
    Epoch 1: Loss: 0.7796, Val Loss: 0.5990, Val Accuracy: 69.67%
    Saving new best model with accuracy: 69.67%
    ...
    Test Accuracy: 76.00%
    ```

- **Explanation Output**:
    ```
    Enter the index of the row you want to explain: 56
    Name  Contribution  Causal Effects
    0    checking_status   -0.115669    0.623556
    ...
    Predicted class: 0
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Captum](https://captum.ai/)
- [SHAP](https://github.com/slundberg/shap)

