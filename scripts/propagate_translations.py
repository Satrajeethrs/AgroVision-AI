#!/usr/bin/env python3
"""
Script to propagate translation keys from Kannada (reference) to all other languages.
For missing translations, it uses English text as placeholder (to be translated manually or via API).
"""

import json
from pathlib import Path

# Language codes to propagate to (excluding English and Kannada which are complete)
TARGET_LANGS = ['hi', 'ta', 'te', 'bn', 'mr', 'ml', 'gu', 'pa', 'or', 'as']

def load_translations():
    """Load the translations file."""
    file_path = Path(__file__).parent.parent / 'translations' / 'messages.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_translations(data):
    """Save translations back to file with pretty formatting."""
    file_path = Path(__file__).parent.parent / 'translations' / 'messages.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved translations to {file_path}")

def propagate_keys():
    """Propagate all keys from Kannada and English to target languages."""
    print("📂 Loading translations...")
    translations = load_translations()
    
    # Get all keys from English (complete reference)
    en_keys = set(translations['en'].keys())
    kn_keys = set(translations['kn'].keys())
    
    # Combine all keys that should exist
    all_keys = en_keys | kn_keys
    print(f"📊 Total unique keys to propagate: {len(all_keys)}")
    
    for lang in TARGET_LANGS:
        print(f"\n🔄 Processing {lang}...")
        
        if lang not in translations:
            translations[lang] = {"_comment": f"{lang} translations"}
        
        existing_keys = set(translations[lang].keys())
        missing_keys = all_keys - existing_keys
        
        if not missing_keys:
            print(f"  ✅ {lang}: All keys present ({len(existing_keys)} keys)")
            continue
        
        print(f"  📝 {lang}: Adding {len(missing_keys)} missing keys...")
        
        added_count = 0
        for key in sorted(missing_keys):
            if key == '_comment':
                continue
            
            # Prefer Kannada translation for reference, fallback to English
            if key in translations['kn']:
                # Use English text as placeholder (will need manual translation)
                ref_text = translations['en'].get(key, f"[TRANSLATE: {key}]")
            else:
                ref_text = translations['en'].get(key, f"[MISSING: {key}]")
            
            translations[lang][key] = ref_text
            added_count += 1
        
        print(f"  ✅ {lang}: Added {added_count} keys (now has {len(translations[lang])} total)")
    
    print("\n💾 Saving updated translations...")
    save_translations(translations)
    
    print("\n" + "="*60)
    print("✅ PROPAGATION COMPLETE!")
    print("="*60)
    print("\n⚠️  NOTE: New keys use English text as placeholders.")
    print("   You can:")
    print("   1. Run the app now - it will show English for new keys")
    print("   2. Use Google Translate API to translate placeholders")
    print("   3. Manually translate important keys later")
    print("\n🌐 Next: Test all languages at http://127.0.0.1:5001")

if __name__ == '__main__':
    propagate_keys()
