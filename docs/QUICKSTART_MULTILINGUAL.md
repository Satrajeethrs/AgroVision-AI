# Quick Start: Multilingual Support

## For End Users

### Step 1: Select Your Language
1. Look for the language dropdown (🌐) in the top right of the header
2. Click and select your preferred language from:
   - English
   - हिन्दी (Hindi)
   - தமிழ் (Tamil)
   - తెలుగు (Telugu)
   - বাংলা (Bengali)
   - मराठी (Marathi)
   - ಕನ್ನಡ (Kannada)
   - മലയാളം (Malayalam)
   - ગુજરાતી (Gujarati)
   - ਪੰਜਾਬੀ (Punjabi)
   - ଓଡ଼ିଆ (Odia)
   - অসমীয়া (Assamese)

3. The page will reload with all content in your language

### Step 2: Use the System
- All form labels, buttons, and instructions are now in your language
- Enter your farm data as usual
- Submit and view results in your language

### Step 3: Print or Save
- Results can be printed in your selected language
- Share recommendations with other farmers in their language

## For Developers

### Quick Setup (5 minutes)

1. **Get API Key** (Optional for testing):
   ```bash
   # Visit https://bhashini.gov.in/ to get your API key
   # For testing, you can skip this - static translations will work
   ```

2. **Configure .env**:
   ```bash
   # Create or edit .env file
   BHASHINI_API_KEY=your_key_here  # Optional
   ENABLE_TRANSLATION=True
   DEFAULT_LANGUAGE=en
   ```

3. **Test Translation**:
   ```bash
   python3 tests/test_translation.py
   ```

4. **Start the App**:
   ```bash
   python3 app.py
   ```

5. **Test in Browser**:
   - Open http://localhost:5001
   - Select a language from dropdown
   - Verify UI is translated

### Adding New Translations

Edit `translations/messages.json`:

```json
{
  "en": {
    "my.new.label": "My Label"
  },
  "hi": {
    "my.new.label": "मेरा लेबल"
  }
}
```

Use in code:

```python
from src.utils.translation import t

label = t('my.new.label', lang='hi')
# Output: "मेरा लेबल"
```

### Translation Modes

#### Mode 1: Static (Works without API key)
- Pre-translated content from JSON
- Instant, no API calls
- 95%+ of UI covered

#### Mode 2: Dynamic (Requires API key)
- Real-time translation via Bhashini
- For user-generated content
- Recommendations and advice

#### Mode 3: LLM Multilingual (Works with any LLM)
- Multilingual prompts to LLM
- AI-generated content in target language
- Best quality for narratives

## Examples

### Example 1: Basic Translation

```python
from src.utils.translation import translate

# Translate a simple string
hindi_text = translate("Apply nitrogen fertilizer", target_lang='hi')
print(hindi_text)
# Output: "नाइट्रोजन उर्वरक लगाएं" (with API key)
# Output: "Apply nitrogen fertilizer" (without API key)
```

### Example 2: Using Translation Keys

```python
from src.utils.translation import t

# Get pre-translated labels
nitrogen_label = t('ui.nitrogen', 'ta')
print(nitrogen_label)
# Output: "நைட்ரஜன் (N)"

status_low = t('status.low', 'te')
print(status_low)
# Output: "తక్కువ"
```

### Example 3: In Flask Routes

```python
from flask import session
from src.utils.translation import t, translate_text

@app.route('/my-advice')
def my_advice():
    lang = session.get('language', 'en')
    
    # Static translation
    title = t('section.fertilizer_recommendation', lang)
    
    # Dynamic translation
    advice = "Use urea fertilizer at planting time"
    if lang != 'en':
        advice = translate_text(advice, lang)
    
    return render_template('advice.html', title=title, advice=advice)
```

### Example 4: With LLM

```python
from src.utils.llm_validator import generate_recommendation_text

lang = session.get('language', 'en')
result = generate_recommendation_text(
    recs=[{'text': 'Plant rice in monsoon season'}],
    data_summary={'rainfall': 1200, 'ph': 6.5},
    target_language=lang
)
print(result['text'])
# Output in Hindi: "मानसून के मौसम में धान लगाने की सिफारिश की जाती है..."
```

## Troubleshooting

### Issue: Language not changing

**Check:**
1. Clear browser cache
2. Check browser console for errors
3. Verify `/set_language` endpoint works:
   ```bash
   curl -X POST http://localhost:5001/set_language \
     -H "Content-Type: application/json" \
     -d '{"language":"hi"}'
   ```

### Issue: Partial translation

**Solution:**
- Some content requires API key for dynamic translation
- Add `BHASHINI_API_KEY` to `.env`
- Restart the application

### Issue: Slow translation

**Solution:**
- Translation is cached after first use
- Pre-translate common phrases
- Use Redis for production caching

## What Gets Translated?

### ✅ Always Translated (No API Key Needed)
- UI labels and buttons
- Form field names
- Section headers
- Status indicators
- Units (kg/ha, °C, %, mm)
- Error messages

### ✅ Translated with API Key
- Crop recommendations
- Fertilizer advice
- Soil management tips
- Environmental analysis
- Action items

### ✅ Translated with LLM
- AI narratives
- Alternative suggestions
- Validation notes
- Custom insights

## Production Deployment

### Performance Tips

1. **Enable Redis caching**:
   ```bash
   pip install redis
   # Update translation.py to use Redis instead of in-memory cache
   ```

2. **Pre-warm cache**:
   ```python
   from src.utils.translation import get_translation_service
   service = get_translation_service()
   
   # Pre-translate common phrases
   common_phrases = ["Low", "Medium", "High", "Optimal", ...]
   for phrase in common_phrases:
       for lang in ['hi', 'ta', 'te']:
           service.translate_text(phrase, 'en', lang)
   ```

3. **Monitor metrics**:
   ```python
   stats = service.get_cache_stats()
   print(f"Cache hit rate: {stats['hit_rate']}")
   ```

### Security

- Store API keys in environment variables
- Never commit `.env` to version control
- Use HTTPS for API calls
- Validate language codes from user input

## Support

- **Documentation**: [docs/MULTILINGUAL.md](docs/MULTILINGUAL.md)
- **Tests**: `tests/test_translation.py`
- **Issues**: [GitHub Issues](https://github.com/Satrajeeth/AgroVision-AI/issues)

---

**Congratulations!** 🎉 Your application now speaks 12 Indian languages!
