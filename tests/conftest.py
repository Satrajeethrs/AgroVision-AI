import pytest
import os
import pickle
from app import app as flask_app


@pytest.fixture(scope='session')
def model(request):
    # Try to load model.pkl if present; otherwise skip tests that require it
    path = os.path.join(request.config.rootpath, 'model.pkl')
    if not os.path.exists(path):
        pytest.skip('model.pkl not present; skipping model-dependent tests')
    with open(path, 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope='session')
def scaler(request):
    path = os.path.join(request.config.rootpath, 'scaler.pkl')
    if not os.path.exists(path):
        pytest.skip('scaler.pkl not present; skipping scaler-dependent tests')
    with open(path, 'rb') as f:
        return pickle.load(f)


@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client
