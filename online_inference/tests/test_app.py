from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_get_main():
    with TestClient(app) as client_:
        response = client_.get("/")
        assert response.status_code == 200
        assert response.json() == "it is entry point of our predictor"


def test_read_predict():
    response = client.get(
        "http://127.0.0.1:8000/predict/",
        json={"data": [[43.0, 0.0, 2.0, 150.0, 256.0, 0.0, 0.0, 182.0, 0.0, 1.8, 0.0, 2.0, 2.0]],
              "features": ["age", "sex", "cp", "trestbps", "chol", "fbs",
                           "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]},
    )
    assert response.status_code == 200
    assert response.json()[0]["target"] == 1
