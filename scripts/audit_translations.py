#!/usr/bin/env python3
"""
Audit translations to find keys that are missing or still in English
for each non-English language. Generates a concise Markdown report and
an optional JSON summary for tooling.

Usage:
  python3 scripts/audit_translations.py [--json]

Outputs:
  - docs/TRANSLATION_AUDIT.md (human readable)
  - translations/missing_by_lang.json (when --json passed)
"""
from __future__ import annotations
import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
TRANSLATIONS = ROOT / "translations" / "messages.json"
REPORT_MD = ROOT / "docs" / "TRANSLATION_AUDIT.md"
MISSING_JSON = ROOT / "translations" / "missing_by_lang.json"

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
}

ASCII_RE = re.compile(r"^[\x00-\x7F]+$")


def looks_english(s: str) -> bool:
    if not s or not isinstance(s, str):
        return True
    # If it contains letters and is ASCII-only, treat as English
    has_alpha = any(ch.isalpha() for ch in s)
    return has_alpha and bool(ASCII_RE.match(s))


def load_translations() -> Dict:
    with open(TRANSLATIONS, "r", encoding="utf-8") as f:
        return json.load(f)


def audit(trans: Dict) -> Tuple[Dict[str, List[Dict]], Dict[str, int]]:
    en = trans.get("en", {})
    all_keys = set(en.keys())
    missing: Dict[str, List[Dict]] = {}
    counts: Dict[str, int] = {}

    for lang, d in trans.items():
        if lang == "en":
            continue
        issues: List[Dict] = []
        for k in sorted(all_keys):
            src = en.get(k, "")
            tgt = d.get(k)
            if tgt is None or tgt.strip() == "":
                issues.append({"key": k, "reason": "missing", "en": src, "tgt": tgt})
                continue
            # identical to English or looks English only
            if tgt.strip() == src.strip() or looks_english(tgt):
                issues.append({"key": k, "reason": "english_placeholder", "en": src, "tgt": tgt})
        missing[lang] = issues
        counts[lang] = len(issues)
    return missing, counts


def write_markdown(missing: Dict[str, List[Dict]], counts: Dict[str, int]) -> None:
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    total = sum(counts.values())
    lines = []
    lines.append("# Translation Audit\n")
    lines.append(f"Total outstanding items: {total}\n")
    lines.append("")

    # Summary table
    lines.append("## Summary by Language\n")
    lines.append("| Language | Missing/English Keys |\n|---|---:|")
    for lang in sorted(missing.keys()):
        name = LANG_NAMES.get(lang, lang)
        lines.append(f"| {name} ({lang}) | {counts.get(lang, 0)} |")
    lines.append("")

    # Top offenders preview
    for lang in sorted(missing.keys()):
        issues = missing[lang]
        if not issues:
            continue
        name = LANG_NAMES.get(lang, lang)
        lines.append(f"## {name} ({lang})\n")
        preview = issues[:25]
        lines.append("| Key | Reason | English | Current |\n|---|---|---|---|")
        for item in preview:
            en_txt = (item.get("en") or "").replace("\n", " ")
            tgt_txt = (item.get("tgt") or "").replace("\n", " ")
            lines.append(f"| `{item['key']}` | {item['reason']} | {en_txt[:60]} | {tgt_txt[:60]} |")
        if len(issues) > len(preview):
            lines.append(f"… and {len(issues) - len(preview)} more.\n")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    trans = load_translations()
    missing, counts = audit(trans)
    write_markdown(missing, counts)
    if "--json" in sys.argv:
        MISSING_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(MISSING_JSON, "w", encoding="utf-8") as f:
            json.dump(missing, f, ensure_ascii=False, indent=2)
    # Console summary
    print("\n=== Translation Audit Complete ===")
    for lang, cnt in sorted(counts.items()):
        print(f"{LANG_NAMES.get(lang, lang)} ({lang}): {cnt} items")
    print(f"Report: {REPORT_MD}")
    if "--json" in sys.argv:
        print(f"JSON:   {MISSING_JSON}")


if __name__ == "__main__":
    main()
