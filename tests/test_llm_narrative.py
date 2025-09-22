import os
from llm_validator import generate_recommendation_text, detect_provider


def test_generate_recommendation_text_uses_stub_when_no_env():
    # Ensure env vars not set
    os.environ.pop('GEMINI_API_KEY', None)
    os.environ.pop('LMSTUDIO_API_KEY', None)
    os.environ.pop('LMSTUDIO_BASE_URL', None)
    out = generate_recommendation_text([{'id':'r1','text':'Apply X'}], {'soil_n':10})
    assert isinstance(out, dict)
    assert 'text' in out
    assert out.get('provider') in ('stub', None)


def test_placeholder_values_ignored():
    # Set a short placeholder-like GEMINI key and ensure it's ignored
    os.environ['GEMINI_API_KEY'] = 'example_key'
    os.environ.pop('FORCE_LLM_PROVIDER', None)
    cfg = detect_provider()
    assert cfg.name == 'stub'
    os.environ.pop('GEMINI_API_KEY', None)


def test_force_provider_accepts_placeholder():
    os.environ['GEMINI_API_KEY'] = 'example_key'
    os.environ['FORCE_LLM_PROVIDER'] = 'true'
    cfg = detect_provider()
    assert cfg.name == 'gemini'
    os.environ.pop('GEMINI_API_KEY', None)
    os.environ.pop('FORCE_LLM_PROVIDER', None)
