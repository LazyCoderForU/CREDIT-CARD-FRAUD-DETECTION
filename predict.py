import kagglehub
import pickle
import numpy as np

# Download latest version
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)

# Load the saved model
with open('rf_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Example input to test the loaded model
example_input = np.array([[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, 0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, 0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0]])

# Make a prediction
prediction = loaded_model.predict(example_input)
print(f"Prediction for the example input: {prediction}")