import os
import pytest
from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


def test_analyze_and_results_shows_ai_narrative(client):
    # Prepare valid form data (within input_validation ranges)
    data = {
        'N': '60', 'P': '40', 'K': '30', 'temperature': '25', 'humidity': '60', 'ph': '6.5', 'rainfall': '800'
    }
    # POST to /analyze
    resp = client.post('/analyze', data=data, follow_redirects=True)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # Ensure the AI Narrative heading or block appears
    assert 'AI Narrative' in html or 'ai-narrative' in html
