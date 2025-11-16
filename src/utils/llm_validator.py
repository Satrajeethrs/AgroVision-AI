"""
Lightweight LLM-backed recommendation validator.

Behavior:
- Reads `.env` if present and detects provider by keys:
  - Gemini: GEMINI_API_KEY or GEMINI_KEY
  - LM Studio: LMSTUDIO_API_KEY or LM_STUDIO_KEY
- If provider key present, will attempt a request (simple HTTP wrapper).
- If no key, will use a local deterministic stub for offline testing.

Public API:
- validate_recommendations(recs, data_summary, top_k=5) -> dict

The function returns a structured dictionary with validation results ready for display.
"""
from __future__ import annotations
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv optional
    pass

import requests


@dataclass
class ProviderConfig:
    name: str
    api_key: Optional[str]
    base_url: Optional[str]


def detect_provider() -> ProviderConfig:
    # Prefer LM Studio if present; many users run a local LM Studio server
    lm_keys = ["LMSTUDIO_API_KEY", "LM_STUDIO_KEY", "LMSTUDIO_KEY"]
    lm_base = os.getenv("LMSTUDIO_BASE_URL")
    lm_model = os.getenv("LMSTUDIO_MODEL_NAME")
    # If an API key exists, use it; otherwise prefer LM Studio if base URL or model name is provided
    for k in lm_keys:
        v = os.getenv(k)
        if v:
            force = os.getenv("FORCE_LLM_PROVIDER", "").lower() in ("1", "true", "yes")
            if not force:
                if is_placeholder_value(v):
                    continue
            return ProviderConfig(name="lmstudio", api_key=v, base_url=lm_base or "http://localhost:8080")
    if lm_base or lm_model:
        return ProviderConfig(name="lmstudio", api_key=None, base_url=lm_base or "http://localhost:8080")

    # Fallback to Gemini if LM Studio isn't configured
    gemini_keys = ["GEMINI_API_KEY", "GEMINI_KEY"]
    for k in gemini_keys:
        v = os.getenv(k)
        if v:
            # Allow an override to force using the provider even if the value looks like a placeholder
            force = os.getenv("FORCE_LLM_PROVIDER", "").lower() in ("1", "true", "yes")
            if not force:
                # Heuristic: treat obviously placeholder/example values as unset
                if is_placeholder_value(v):
                    continue
            return ProviderConfig(name="gemini", api_key=v, base_url=os.getenv("GEMINI_BASE_URL"))

    return ProviderConfig(name="stub", api_key=None, base_url=None)


def is_placeholder_value(val: str) -> bool:
    """Heuristic check for placeholder/example values in env vars.

    We consider values containing common placeholder substrings or short values as placeholders.
    This is a conservative heuristic to avoid accidentally using example keys included in
    a repository's `.env` file. If you want to force provider usage, set
    `FORCE_LLM_PROVIDER=true` in the environment.
    """
    if not val:
        return True
    v = val.strip().lower()
    placeholder_substrings = ["example", "your_", "xxxx", "xxx", "test", "replace", "dummy"]
    for s in placeholder_substrings:
        if s in v:
            return True
    # very short values unlikely to be real API keys
    if len(v) < 20:
        return True
    return False


def _call_gemini(prompt: str, cfg: ProviderConfig) -> str:
    # Minimal wrapper: Gemini API endpoints vary; we use REST-style interface if provided.
    # If no base_url provided, we can't call real Gemini here.
    if not cfg.base_url:
        return _stub_response(prompt)
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "max_tokens": 512}
    try:
        r = requests.post(cfg.base_url.rstrip("/") + "/v1/generate", headers=headers, json=payload, timeout=10)
        r.raise_for_status()
        return r.json().get("text", r.text)
    except Exception as e:
        return f"[error] {e}"


def _call_lmstudio(prompt: str, cfg: ProviderConfig) -> str:
    # LM Studio deployments vary. Try several common endpoints and payload shapes
    base = (cfg.base_url or os.getenv("LMSTUDIO_BASE_URL") or "http://localhost:8080").rstrip("/")
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    model_name = os.getenv("LMSTUDIO_MODEL_NAME")

    # Candidate endpoints to try (ordered by most common)
    endpoints = []
    if model_name:
        endpoints.append(f"{base}/v1/models/{model_name}:predict")
        endpoints.append(f"{base}/api/models/{model_name}/predict")
        # OpenAI-compatible model-specific endpoints
        endpoints.append(f"{base}/v1/models/{model_name}/chat/completions")
        endpoints.append(f"{base}/v1/models/{model_name}/completions")
    endpoints.extend([
        f"{base}/v1/predict",
        f"{base}/api/predict",
        f"{base}/predict",
        f"{base}/v1/generate",
        # OpenAI-compatible generic endpoints
        f"{base}/v1/chat/completions",
        f"{base}/v1/completions",
    ])

    # Candidate payload shapes
    # Basic payloads plus OpenAI-compatible shapes (include model when available)
    payloads = [
        {"prompt": prompt, "max_output_tokens": 512},
        {"input": prompt, "max_output_tokens": 512},
        {"inputs": [prompt], "max_output_tokens": 512},
        {"data": {"prompt": prompt}, "max_output_tokens": 512},
    ]
    # Add OpenAI-style payloads that many LM Studio deployments accept
    if model_name:
        payloads.append({"model": model_name, "messages": [{"role": "user", "content": prompt}], "max_tokens": 512})
        payloads.append({"model": model_name, "prompt": prompt, "max_tokens": 512})
    else:
        payloads.append({"messages": [{"role": "user", "content": prompt}], "max_tokens": 512})
        payloads.append({"prompt": prompt, "max_tokens": 512})

    last_err = None
    attempts = []
    for url in endpoints:
        for payload in payloads:
            try:
                attempts.append({"url": url, "payload_sample": list(payload.keys())[:3]})
                r = requests.post(url, headers=headers, json=payload, timeout=8)
                # Treat non-2xx as an informative error but capture body
                if r.status_code >= 400:
                    last_err = f"{r.status_code} {r.reason}: {r.text[:400]}"
                    continue
                # Try to decode JSON and handle common shapes (including OpenAI-like responses)
                try:
                    data = r.json()
                except Exception:
                    return r.text

                if isinstance(data, dict):
                    # If the server returned an explicit error that implies wrong endpoint/method,
                    # treat it as a non-terminal failure and continue trying other endpoints.
                    err_text = None
                    for k in ("error", "message", "detail", "msg"):
                        if k in data and isinstance(data[k], str):
                            err_text = data[k]
                            break
                    if err_text:
                        low = err_text.lower()
                        if any(sub in low for sub in ("unexpected endpoint", "unexpected endpoint or method", "method not allowed", "not found", "invalid endpoint")):
                            last_err = err_text
                            continue

                    # Common LM Studio shapes
                    if "results" in data and isinstance(data["results"], list):
                        return "\n".join(str(x) for x in data["results"])[:4000]
                    if "text" in data:
                        return data["text"]

                    # OpenAI-style: {'choices': [{'message': {'content': ...}}]} or {'choices':[{'text': '...'}]}
                    if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                        first = data["choices"][0]
                        # chat completion shape
                        if isinstance(first, dict) and "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                            return first["message"]["content"]
                        # text completion shape
                        if isinstance(first, dict) and "text" in first:
                            return first["text"]
                        # some have 'message'->'content' nested differently
                        if isinstance(first, dict) and "content" in first:
                            return first["content"]

                    # Some deployments return {'outputs':[{'generated_text':...}, ...]}
                    if isinstance(data.get("outputs"), list) and data["outputs"] and isinstance(data["outputs"][0], dict):
                        out = []
                        for o in data["outputs"]:
                            for k in ("generated_text", "text", "content", "output"):
                                if k in o:
                                    out.append(str(o[k]))
                        if out:
                            return "\n".join(out)[:4000]

                    return json.dumps(data)

                if isinstance(data, list):
                    if all(isinstance(x, str) for x in data):
                        return "\n".join(data)[:4000]
                    return json.dumps(data)

            except Exception as e:
                last_err = str(e)
                continue

    # If we get here, no endpoint succeeded. Return a helpful error mentioning attempts.
    attempt_summary = json.dumps(attempts[:6], indent=None)
    err_msg = f"[lmstudio_error] No working endpoint. attempts={attempt_summary} last_error={last_err}"
    return err_msg


def _stub_response(prompt: str) -> str:
    # Deterministic lightweight "AI" response used when no API key is present.
    # We keep responses short and structured so downstream code can parse them.
    return (
        "VALIDATION:\n"
        "-status: plausible\n"
        "-confidence: 0.65\n"
        "-notes: Recommendation matches common patterns but verify numeric thresholds.\n"
    )


def _call_provider(prompt: str, cfg: ProviderConfig) -> str:
    if cfg.name == "gemini":
        return _call_gemini(prompt, cfg)
    if cfg.name == "lmstudio":
        return _call_lmstudio(prompt, cfg)
    return _stub_response(prompt)


def _build_prompt(recs: List[Dict[str, Any]], data_summary: Dict[str, Any]) -> str:
    # Construct a concise prompt for validation.
    lines = ["You are an expert validator. Assess each recommendation for accuracy, consistency with data, and reliability."
             ,"Return a short structured result for each recommendation with fields: id, verdict (valid/invalid/uncertain), confidence(0-1), notes."]
    lines.append("DATA_SUMMARY:")
    lines.append(json.dumps(data_summary, indent=None))
    lines.append("RECOMMENDATIONS:")
    for r in recs:
        rid = r.get("id") or r.get("recommendation_id") or r.get("name")
        lines.append(f"- id: {rid} | text: {r.get('text') or r.get('recommendation')}")
    lines.append("Answer in compact JSON array of objects.")
    return "\n".join(lines)


def parse_response_text(text: str) -> List[Dict[str, Any]]:
    # Attempt to recover JSON from LLM. If it fails, try to extract simple key-values from stub format.
    text = text.strip()
    # Try JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "results" in parsed and isinstance(parsed["results"], list):
            return parsed["results"]
    except Exception:
        pass

    # Fallback: parse stub format lines
    items = []
    if text.startswith("VALIDATION:"):
        # convert to one item
        d = {}
        for line in text.splitlines()[1:]:
            if not line.strip():
                continue
            if line.strip().startswith("-"):
                k, _, v = line.strip()[1:].partition(":")
                d[k.strip()] = v.strip()
        items.append({"id": "stub", "verdict": d.get("status"), "confidence": float(d.get("confidence", 0)), "notes": d.get("notes")})
        return items

    # Last resort: return entire text as a single uncertain note
    return [{"id": "unknown", "verdict": "uncertain", "confidence": 0.0, "notes": text}]


def validate_recommendations(recs: List[Dict[str, Any]], data_summary: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
    """Validate recommendations using detected LLM provider.

    Returns structure:
    {
      "provider": "gemini|lmstudio|stub",
      "raw": <raw LLM text or error>,
      "results": [ {id, verdict, confidence, notes}, ... ]
    }
    """
    cfg = detect_provider()
    prompt = _build_prompt(recs[:top_k], data_summary)
    raw = _call_provider(prompt, cfg)
    parsed = parse_response_text(raw)
    return {"provider": cfg.name, "raw": raw, "results": parsed}


def generate_recommendation_text(recs: List[Dict[str, Any]], data_summary: Dict[str, Any], top_k: int = 1) -> Dict[str, Any]:
    """Generate a short human-readable recommendation/narrative using the detected provider.

    This is a thin wrapper that builds a compact prompt and returns the raw text response
    (or the stub). It is intentionally simple: callers should render the returned text
    as HTML-safe content after sanitizing or trusting the environment.
    """
    cfg = detect_provider()
    # Reuse a compact prompt pattern
    lines = [
        "You are an expert agronomist. Create one short, clear recommendation paragraph (2-4 sentences) for a farmer based on the data below.",
        "Return plain text only.",
        "DATA_SUMMARY:",
        json.dumps(data_summary, indent=None),
        "RECOMMENDATIONS:",
    ]
    for r in recs[:top_k]:
        rid = r.get("id") or r.get("recommendation_id") or r.get("name")
        lines.append(f"- id: {rid} | text: {r.get('text')}")

    prompt = "\n".join(lines)
    raw = _call_provider(prompt, cfg)
    # If provider returned an error-like string, fall back to stub
    if isinstance(raw, str) and raw.startswith("[error]"):
        fallback = _stub_response(prompt)
        return {"provider": cfg.name, "text": fallback, "raw": raw}
    return {"provider": cfg.name, "text": str(raw), "raw": raw}


def generate_alternative_recommendation(inputs: Dict[str, Any], top_k: int = 1) -> Dict[str, Any]:
    """Ask the LLM for an alternative crop recommendation and a short rationale.

    We request output as JSON when possible: {"crop": "Maize", "rationale": "..."}
    The function attempts to parse JSON; if parsing fails it returns the raw text under 'rationale'.
    """
    cfg = detect_provider()
    prompt_lines = [
        "You are an expert agronomist.",
        "Given the following numeric farm inputs, suggest the most suitable crop (single word or short name) and a brief 1-2 sentence rationale.",
        "Return output as a single JSON object with keys: crop, rationale.",
        "INPUTS:",
        json.dumps(inputs, indent=None)
    ]
    prompt = "\n".join(prompt_lines)
    raw = _call_provider(prompt, cfg)

    # Try to parse JSON from the response
    text = None
    try:
        if isinstance(raw, str):
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and 'crop' in parsed:
                return {'provider': cfg.name, 'crop': parsed.get('crop'), 'rationale': parsed.get('rationale', ''), 'raw': raw}
    except Exception:
        pass

    # If provider returned the validator-style stub, treat as stub (use rule-based fallback)
    raw_text = str(raw) if raw is not None else ''
    if raw_text.strip().startswith('VALIDATION:'):
        cfg = ProviderConfig(name='stub', api_key=None, base_url=None)

    # Fallback: attempt to extract a crop name heuristically
    text = str(raw)
    # crude heuristic: pick the first capitalized word that looks like a crop
    import re
    m = re.search(r"\b([A-Z][a-z]{2,})\b", text)
    crop_guess = m.group(1) if m else None

    # If provider is 'stub' or we couldn't parse any crop, generate a deterministic rule-based alternative
    if cfg.name == 'stub' or not crop_guess or raw_text.strip().startswith('VALIDATION:'):
        # use simple rules based on rainfall and ph if available in inputs raw prompt
        try:
            obj = json.loads(prompt.split('INPUTS:')[-1]) if 'INPUTS:' in prompt else {}
            rainfall = float(obj.get('rainfall') or obj.get('rainfall_mm') or 0)
            ph = float(obj.get('ph') or 6.5)
        except Exception:
            rainfall = 0
            ph = 6.5

        # simple deterministic rules
        if rainfall > 1000 and 6.0 <= ph <= 7.5:
            crop_guess = 'Rice'
            rationale = 'High rainfall and near-neutral pH favor paddy rice.'
        elif rainfall < 400:
            crop_guess = 'Millet'
            rationale = 'Low rainfall suggests drought-tolerant millets.'
        elif ph < 5.5:
            crop_guess = 'Wheat'
            rationale = 'Acidic soils favor crops suited to lower pH; consider wheat varieties and liming.'
        else:
            crop_guess = 'Maize'
            rationale = 'General-purpose cereal suitable for moderate conditions.'

        return {'provider': cfg.name, 'crop': crop_guess, 'rationale': rationale, 'raw': raw}

    return {'provider': cfg.name, 'crop': crop_guess, 'rationale': text, 'raw': raw}


if __name__ == "__main__":
    # Quick manual demo
    sample_recs = [{"id": "r1", "text": "Apply 50 kg/ha of nitrogen at planting."},
                   {"id": "r2", "text": "Irrigate twice per week during fruiting."}]
    sample_data = {"soil_n": 12, "rainfall_mm": 200, "crop": "maize"}
    out = validate_recommendations(sample_recs, sample_data)
    print(json.dumps(out, indent=2))
