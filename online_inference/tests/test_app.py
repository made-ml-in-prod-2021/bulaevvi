import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "..")
from online_inference.app import app


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_app(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == True
    response = client.get("/abracadabra")
    assert response.status_code >= 400
