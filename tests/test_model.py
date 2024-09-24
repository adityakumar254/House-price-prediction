# tests/test_model.py
import unittest
import joblib
import pandas as pd

class TestModel(unittest.TestCase):
    def setUp(self):
        # Load the trained model
        self.model = joblib.load('model/house_price_model.pkl')

    def test_prediction(self):
        # Create sample input data
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
        prediction = self.model.predict(new_data)
        
        # Check if the prediction is a float and reasonable
        self.assertIsInstance(prediction[0], float)

if __name__ == '__main__':
    unittest.main()
