from app import app as flask_app


def test_validate_recs_endpoint(client):
    # Prepare session by calling analyze first
    data = {
        'N': '60', 'P': '40', 'K': '30', 'temperature': '25', 'humidity': '60', 'ph': '6.5', 'rainfall': '800'
    }
    rv = client.post('/analyze', data=data, follow_redirects=True)
    assert rv.status_code == 200
    # Now call validation endpoint
    resp = client.post('/validate_recs')
    assert resp.status_code == 200
    json = resp.get_json()
    assert 'provider' in json
    assert 'results' in json
