from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    r = client.root.get('/')
    assert r.status_code == 200
    assert r.json() == "Welcome!"


def test_predict_positive():
    data = {'age': 52,
            'workclass': 'Self-emp-inc',
            'fnlgt': 287927,
            'education': 'HS-grad',
            'education-num': 9,
            'marital-status': 'Married-civ-spouse',
            'occupation': 'Exec-managerial',
            'relationship': 'Wife',
            'race': 'White',
            'sex': 'Female',
            'capital-gain': 15024,
            'capital-loss': 0,
            'hours-per-week': 40,
            'native-country': 'United-States'
            }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.text == ">50K"


def test_predict_negative():
    data = {'age': 39,
            'workclass': 'State-gov',
            'fnlgt': 77516,
            'education': 'Bachelors',
            'education-num': 13,
            'marital-status': 'Never-married',
            'occupation': 'Adm-clerical',
            'relationship': 'Not-in-family',
            'race': 'White',
            'sex': 'Male',
            'capital-gain': 2174,
            'capital-loss': 0,
            'hours-per-week': 40,
            'native-country': 'United-States'
            }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.text == "<=50K"


def test_predict_invalid():
    data = {}
    response = client.post("/predict", json=data)
    assert response.status_code == 422
