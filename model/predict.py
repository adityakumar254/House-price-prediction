# model/predict.py
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model/house_price_model.pkl')

# Load new data for prediction (this can be changed to new incoming data)
new_data = pd.DataFrame({
    'MedInc': [8.3252],
    'HouseAge': [41.0],
    'AveRooms': [6.984127],
    'AveBedrms': [1.023810],
    'Population': [322.0],
    'AveOccup': [2.555556],
    'Latitude': [37.88],
    'Longitude': [-122.23]
})

# Make predictions
predicted_price = model.predict(new_data)
print(f'Predicted Price: {predicted_price[0]}')
