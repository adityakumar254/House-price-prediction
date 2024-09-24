# tests/test_model.py

import pytest
import pandas as pd
from sklearn.metrics import mean_squared_error
from model.train import model  # Assuming 'model' is defined in train.py

@pytest.fixture
def setup_data():
    data = pd.read_csv('data/california_housing.csv')
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def test_model_training(setup_data):
    X_train, X_test, y_train, y_test = setup_data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert mean_squared_error(y_test, y_pred) < 1.0  # Example assertion, adjust as needed

# Add more tests as necessary
