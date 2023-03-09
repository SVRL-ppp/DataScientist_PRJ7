import sys
from fastapi.testclient import TestClient
from fastapi import status
sys.path.append('api/backend/')

from fast_api import app

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
def test_prediction():
    response = client.post("/predict", json={"SKID":str(100001)})
    assert response.status_code == 200
    # In function of models ! the output can change so ! == here


      def test_settings_passed(self):
        response = self.client.get('/test', follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    def test_settings_failed(self):
        response = self.client.get('/test_not_exist', follow_redirects=True)
        self.assertEqual(response.status_code, 404)
