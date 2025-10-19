from fastapi.testclient import TestClient
from main import app

def test_health():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True

def test_meta():
    c = TestClient(app)
    r = c.get("/meta")
    j = r.json()
    assert r.status_code == 200
    for k in ["model","model_version","git_sha","last_updated","trained","weights_format","weights_uri","runtime"]:
        assert k in j

def test_predict_shape():
    c = TestClient(app)
    r = c.post("/predict", json={"params": {}, "data": {}})
    assert r.status_code in (200, 500, 422)
    if r.status_code == 200:
        j = r.json()
        assert j["ok"] is True
        for k in ["trace_id","model","model_version","timing_ms","result"]:
            assert k in j
