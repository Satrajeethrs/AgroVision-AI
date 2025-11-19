#!/usr/bin/env python3
"""
Backfill static translations using IndicTrans2 for keys that are
missing or that still contain English placeholders.

Run inside Docker (models available):
  docker-compose exec agrovision python3 scripts/backfill_translations.py

Then copy updated file back to host:
  docker cp agrovision-ai:/app/translations/messages.json translations/messages.json

Options:
  --langs hi,ta,...   Comma-separated list of target languages (default: all non-English)
  --limit N           Limit number of keys per language (for quick runs)
  --dry-run           Do not write changes, only print summary
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from datetime import datetime
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

# Project paths
ROOT = Path(__file__).resolve().parents[1]
TRANSLATIONS = ROOT / "translations" / "messages.json"
BACKUP_DIR = ROOT / "translations" / "backups"
LOG_DIR = ROOT / "logs"

# Prefer project translation utilities
sys.path.insert(0, str(ROOT))
from src.utils.translation import translate  # noqa: E402

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
PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")  # matches {name}


def looks_english(s: str) -> bool:
    if not s or not isinstance(s, str):
        return True
    has_alpha = any(ch.isalpha() for ch in s)
    return has_alpha and bool(ASCII_RE.match(s))


def protect_placeholders(text: str) -> Tuple[str, List[str]]:
    """Replace {placeholders} with sentinels to avoid being translated.
    Returns protected text and list of original placeholder tokens in order.
    """
    placeholders = PLACEHOLDER_RE.findall(text)
    protected = text
    for i, ph in enumerate(placeholders):
        protected = protected.replace(ph, f"__PH_{i}__")
    return protected, placeholders


def restore_placeholders(text: str, placeholders: List[str]) -> str:
    out = text
    for i, ph in enumerate(placeholders):
        out = out.replace(f"__PH_{i}__", ph)
    return out


def load_translations() -> Dict:
    with open(TRANSLATIONS, "r", encoding="utf-8") as f:
        return json.load(f)


def save_translations(data: Dict) -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"messages.backup.{ts}.json"
    # Backup
    with open(backup, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # Write main file
    with open(TRANSLATIONS, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _matches_any_prefix(key: str, prefixes: Iterable[str]) -> bool:
    return any(key.startswith(pfx) for pfx in prefixes)


def plan_backfill(
    data: Dict,
    targets: List[str],
    only_prefixes: List[str] | None = None,
    priority_prefixes: List[str] | None = None,
) -> Dict[str, List[str]]:
    """Return mapping lang -> list of keys to backfill.

    - only_prefixes: if provided, consider only keys starting with any of these prefixes
    - priority_prefixes: if provided, sort selected keys so these prefixes come first
    """
    en = data.get("en", {})
    all_keys = sorted(en.keys())
    if only_prefixes:
        all_keys = [k for k in all_keys if _matches_any_prefix(k, only_prefixes)]

    plan: Dict[str, List[str]] = {}
    for lang in targets:
        cur = data.get(lang, {})
        todo_basic: List[str] = []
        for k in all_keys:
            src = en.get(k, "")
            tgt = cur.get(k)
            if tgt is None or (isinstance(tgt, str) and tgt.strip() == ""):
                todo_basic.append(k)
                continue
            if isinstance(tgt, str) and (tgt.strip() == str(src).strip() or looks_english(tgt)):
                todo_basic.append(k)

        # Reorder by priority prefixes if provided
        if priority_prefixes:
            pri = [k for k in todo_basic if _matches_any_prefix(k, priority_prefixes)]
            rest = [k for k in todo_basic if k not in pri]
            todo = pri + rest
        else:
            todo = todo_basic

        plan[lang] = todo
    return plan


def do_backfill(
    data: Dict,
    plan: Dict[str, List[str]],
    limit: int | None = None,
    *,
    verbose: bool = False,
    log_fn=None,
    echo_interval: int = 10,
) -> Dict[str, int]:
    en = data.get("en", {})
    changed: Dict[str, int] = {lang: 0 for lang in plan.keys()}

    for lang, keys in plan.items():
        if not keys:
            continue
        total = len(keys)
        if limit is not None:
            total = min(total, limit)
        if verbose:
            print(f"[backfill] {lang}: {len(keys)} planned, limit={limit}")
        if log_fn:
            log_fn(f"START lang={lang} planned={len(keys)} limit={limit}")

        count = 0
        for k in keys:
            if limit is not None and count >= limit:
                break
            src = en.get(k)
            if not isinstance(src, str) or not src.strip():
                continue
            # Protect placeholders
            protected, phs = protect_placeholders(src)
            # Translate
            translated = translate(protected, target_lang=lang, source_lang='en')
            if not translated or not isinstance(translated, str):
                continue
            restored = restore_placeholders(translated, phs)
            # Save only if non-empty
            data[lang][k] = restored
            if log_fn:
                log_fn(f"OK lang={lang} key={k}")
            count += 1
            if verbose and count % echo_interval == 0:
                print(f"[backfill] {lang}: {count}/{total} done...")
        if verbose:
            print(f"[backfill] {lang}: completed {count} updates")
        if log_fn:
            log_fn(f"END lang={lang} updated={count}")
        changed[lang] = count
    return changed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--langs", type=str, default="",
                   help="Comma-separated languages (default: all non-English)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit keys per language (for quick test)")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not write changes, only print summary")
    p.add_argument("--only-prefixes", type=str, default="",
                   help="Comma-separated key prefixes to restrict backfill (e.g., ui.,nav.,section.)")
    p.add_argument("--priority-prefixes", type=str, default="",
                   help="Comma-separated key prefixes to prioritize first in backfill order")
    p.add_argument("--verbose", action="store_true",
                   help="Print progress for each language and periodic updates")
    p.add_argument("--log-file", type=str, default=str(LOG_DIR / "translation_backfill.log"),
                   help="Path to append detailed backfill logs")
    p.add_argument("--echo-interval", type=int, default=10,
                   help="Print a progress line every N keys when --verbose is enabled")
    return p.parse_args()


def main():
    args = parse_args()
    data = load_translations()
    all_langs = [l for l in data.keys() if l != 'en']
    targets = [s.strip() for s in args.langs.split(',') if s.strip()] if args.langs else all_langs
    only_prefixes = [s.strip() for s in args.only_prefixes.split(',') if s.strip()]
    priority_prefixes = [s.strip() for s in args.priority_prefixes.split(',') if s.strip()]

    # Prepare logger
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def log_fn(msg: str):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_path, 'a', encoding='utf-8') as lf:
            lf.write(f"{ts} | {msg}\n")

    # Ensure language sections exist
    for lang in targets:
        data.setdefault(lang, {})

    plan = plan_backfill(
        data,
        targets,
        only_prefixes=only_prefixes or None,
        priority_prefixes=priority_prefixes or None,
    )
    total_todo = sum(len(v) for v in plan.values())
    print(f"Planned backfill items: {total_todo}")
    for lang in targets:
        print(f"  {LANG_NAMES.get(lang, lang)} ({lang}): {len(plan[lang])}")

    if args.dry_run:
        print("Dry run. No changes written.")
        return

    changed = do_backfill(
        data,
        plan,
        limit=args.limit,
        verbose=args.verbose,
        log_fn=log_fn,
        echo_interval=args.echo_interval,
    )
    total_changed = sum(changed.values())
    print("\nApplied changes:")
    for lang in targets:
        print(f"  {LANG_NAMES.get(lang, lang)} ({lang}): {changed.get(lang,0)} updated")
    print(f"Total updated: {total_changed}")

    if total_changed > 0:
        save_translations(data)
        print(f"\nSaved updated translations to {TRANSLATIONS}")
    else:
        print("\nNo updates necessary.")


if __name__ == "__main__":
    main()
