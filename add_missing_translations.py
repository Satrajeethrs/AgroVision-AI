"""Script to add missing translation keys to all language sections"""
import json
from src.utils.translation import translate

# Load existing translations
with open('translations/messages.json', 'r', encoding='utf-8') as f:
    translations = json.load(f)

# Get all keys from English section
english_keys = set(translations['en'].keys())

# Languages to update
languages = ['hi', 'ta', 'te', 'bn', 'mr', 'kn', 'ml', 'gu', 'pa', 'or', 'as']
lang_codes_full = {
    'hi': 'hin_Deva',
    'ta': 'tam_Taml',
    'te': 'tel_Telu',
    'bn': 'ben_Beng',
    'mr': 'mar_Deva',
    'kn': 'kan_Knda',
    'ml': 'mal_Mlym',
    'gu': 'guj_Gujr',
    'pa': 'pan_Guru',
    'or': 'ory_Orya',
    'as': 'asm_Beng'
}

print("Adding missing translation keys to all languages...")

for lang in languages:
    print(f"\nProcessing {lang}...")
    
    # Get existing keys for this language
    existing_keys = set(translations[lang].keys())
    
    # Find missing keys
    missing_keys = english_keys - existing_keys
    
    if not missing_keys:
        print(f"  ✓ No missing keys for {lang}")
        continue
    
    print(f"  Found {len(missing_keys)} missing keys")
    
    # Add missing keys with translations
    for key in sorted(missing_keys):
        if key == '_comment':
            continue
            
        english_text = translations['en'][key]
        
        # Translate from English to target language
        try:
            translated_text = translate(english_text, lang, 'en')
            translations[lang][key] = translated_text
            print(f"    Added: {key} = {translated_text}")
        except Exception as e:
            print(f"    Error translating {key}: {e}")
            # Fallback to English if translation fails
            translations[lang][key] = english_text

# Save updated translations
with open('translations/messages.json', 'w', encoding='utf-8') as f:
    json.dump(translations, f, ensure_ascii=False, indent=2)

print("\n✅ All missing translation keys have been added!")
