import sys
from fastapi.testclient import TestClient
from fastapi import status
sys.path.append('api/backend/')

from fast_api import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"health_check":"OK"}

def test_prediction():
    response = client.post("/predict", json={"SKID":"100001"})
    assert response.status_code == 200