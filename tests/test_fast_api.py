import sys
from fastapi.testclient import TestClient
from fastapi import status
sys.path.append('fastapi')

from fast_api import app
import json

client = TestClient(app)

# Test communication with API
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"health_check":"OK"}

# test retour des code erreurs
def test_error():
    response = client.post("/predict_don'texist",  follow_redirects=True)
    assert response.status_code == 404
    
# Test of API prediction format
def test_prediction_fake_ID():
    response = client.post("/predict", json={'SKID': str(100001)}, follow_redirects=False) # json={"SKID":"100001"}
    assert response.status_code == 400

