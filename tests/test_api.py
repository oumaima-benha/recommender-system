from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_recommend():
    response = client.get("/recommend/1")
    
    assert response.status_code == 200
    
    data = response.json()
    
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)


def test_similar():
    response = client.get("/similar/Star%20Wars%20(1977)")
    
    assert response.status_code == 200
    
    data = response.json()
    
    assert "similar_movies" in data