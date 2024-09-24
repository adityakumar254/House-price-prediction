import pandas as pd
import joblib

# Load the model
model = joblib.load('model/california_housing_model.pkl')

# Example of making a prediction
def predict(input_data):
    # Assume input_data is a pandas DataFrame similar to X_test
    predictions = model.predict(input_data)
    return predictions

# Example input data for testing
input_data = pd.DataFrame({
    'longitude': [-122.23, -122.22],
    'latitude': [37.88, 37.89],
    'housing_median_age': [41, 21],
    'total_rooms': [880, 712],
    'total_bedrooms': [129, 110],
    'population': [322, 240],
    'households': [126, 97],
    'median_income': [8.3252, 8.3014],
})

# Make prediction
predictions = predict(input_data)
print("Predictions:", predictions)
