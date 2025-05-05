# Credit Card Fraud Detection

## Overview
This repository contains a machine learning project aimed at detecting fraudulent credit card transactions. The project includes data preprocessing, handling class imbalance, training classification models, and evaluating their performance. It also provides tools for downloading datasets and making predictions using a trained model.

## Features
- **Data Preprocessing**: Normalization and scaling of transaction data.
- **Class Imbalance Handling**: Oversampling using SMOTE.
- **Model Training**: Logistic Regression and Random Forest classifiers.
- **Model Evaluation**: Metrics like precision, recall, and F1-score.
- **Visualization**: Class distribution, feature correlations, and feature distributions.
- **Model Persistence**: Save and load trained models using pickle.
- **Dataset Download**: Automated dataset download using Kaggle API.

## Repository Structure
```
CREDIT-CARD-FRAUD-DETECTION
├── 1.py                     # Script to load and test the trained model
├── credit_card_fraud_detection.ipynb  # Jupyter Notebook for the entire workflow
├── download_dataset.py      # Script to download the dataset from Kaggle
├── predict.py               # Script for making predictions using the trained model
├── README.md                # Project documentation
├── rf_model.pkl             # Trained Random Forest model
```

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Required Python libraries:
  - pandas
  - numpy
  - scikit-learn
  - imbalanced-learn
  - matplotlib
  - seaborn
  - kagglehub

Install the required libraries using:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn kagglehub
```

### Dataset
The dataset is automatically downloaded using the `download_dataset.py` script. Ensure you have Kaggle API credentials set up before running the script.

Run the following command to download the dataset:
```bash
python download_dataset.py
```

### Training the Model
Use the Jupyter Notebook `credit_card_fraud_detection.ipynb` to:
1. Preprocess the data.
2. Handle class imbalance.
3. Train the model.
4. Evaluate the model.

### Making Predictions
Use the `1.py` script to load the trained model and make predictions on new data:
```bash
python 1.py
```

### Visualization
The notebook includes visualizations for:
- Class distribution.
- Feature correlations.
- Feature distributions.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Dataset: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Libraries: scikit-learn, imbalanced-learn, matplotlib, seaborn, kagglehub
