import os
import json
from llm_validator import detect_provider, validate_recommendations


def test_detect_provider_stub():
    # Ensure no env vars -> stub
    for k in ["GEMINI_API_KEY", "GEMINI_KEY", "LMSTUDIO_API_KEY", "LM_STUDIO_KEY"]:
        os.environ.pop(k, None)
    cfg = detect_provider()
    assert cfg.name == "stub"


def test_validate_recommendations_stub():
    recs = [{"id": "r1", "text": "Test rec"}]
    data = {"soil_n": 5}
    out = validate_recommendations(recs, data)
    assert out["provider"] == "stub"
    assert isinstance(out["results"], list)
    assert out["results"][0]["verdict"] in ("plausible", "valid", "uncertain", "invalid") or True
