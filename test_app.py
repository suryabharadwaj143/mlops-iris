from fastapi.testclient import TestClient
from main import app

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong"}


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
        assert response.json() == {"flower_class": "Iris Virginica"}

#Task 2 Writing Test Cases
# test to check the correct functioning of the /ping route
def test_iris():
    with TestClient(app) as client:
        response = client.get("/iris")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"iris": "This is a iris flower prediction website"}

# test to check the correct functioning of the /ping route
def test_prediction():
    with TestClient(app) as client:
        response = client.get("/prediction")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"prediction": "We predicted our test cases"}