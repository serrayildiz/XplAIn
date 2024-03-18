# Explainable AI Project

This project demonstrates the use of explainable AI techniques to interpret the predictions of machine learning models. It includes examples using different datasets from the finance and healthcare domains.

## Getting Started

To run the examples in this project, you will need Python installed on your machine along with the following libraries:
- pandas
- scikit-learn
- lime

You can install these libraries using pip:

```bash
pip install pandas scikit-learn lime
```

## Datasets

The project includes examples using the following datasets:
- Loan Approval Dataset (Finance)
- Breast Cancer Dataset (Healthcare)
- Iris Dataset (General)

## Usage

To run the examples, execute the corresponding Python script for each dataset:

- For the Loan Approval dataset:

  ```bash
  python finance.py
  ```

- For the Breast Cancer dataset:

  ```bash
  python breast.py
  ```

- For the Iris dataset:

  ```bash
  python iris.py
  ```

Each script will train a machine learning model on the respective dataset and use LIME (Local Interpretable Model-agnostic Explanations) to explain the predictions of the model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
