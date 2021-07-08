#Importing required libraries
from fastapi.testclient import TestClient
from main import app
from datetime import datetime

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        # Printing Time Stamp and Response
        assert response.json() == {"ping": "pong" , "timestamp": datetime.now()}


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 3,
        "sepal_width": 5,
        "petal_length": 3.2,
        "petal_width": 4.4,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
         # Printing Time Stamp and Response
        assert response.json() == {"flower_class": "Iris Virginica", "timestamp": datetime.now()}

#Task2 : Writing test cases 

# Test to check the correct functioning of the /iris route
def test_iris():
    with TestClient(app) as client:
        response = client.get("/iris")
        # asserting the correct response is received
        assert response.status_code == 200
         # Printing Time Stamp and Response
        assert response.json() == {"iris": "This is a Iris Flower prediction page", "timestamp": datetime.now()}


# Test to check the correct functioning of the /predction route
def test_predction():
    with TestClient(app) as client:
        response = client.get("/predction")
        # asserting the correct response is received
        assert response.status_code == 200
         # Printing Time Stamp and Response
        assert response.json() == {"predction": "Iris flower predcited successfully", "timestamp": datetime.datetime.now()}

        
