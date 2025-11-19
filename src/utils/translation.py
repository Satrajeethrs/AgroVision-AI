"""
Translation utilities for AgroVision-AI multilingual support using AI4Bharat's IndicTrans2.

This module provides translation services using open-source IndicTrans2 models for
dynamic content translation, combined with static translations from JSON files
for UI elements.

Features:
- Multilingual support for 12 Indian languages
- Static translations for UI elements (messages.json)
- Dynamic translation via IndicTrans2 open-source models (NO API KEY REQUIRED)
- Translation caching for performance
- Graceful degradation when models are not installed
- Session-based language persistence
- Offline translation capability

Supported Languages:
- English (en), Hindi (hi), Tamil (ta), Telugu (te)
- Bengali (bn), Marathi (mr), Kannada (kn), Malayalam (ml)
- Gujarati (gu), Punjabi (pa), Odia (or), Assamese (as)

IndicTrans2 Models:
- ai4bharat/indictrans2-en-indic-1B (English to Indic languages)
- ai4bharat/indictrans2-indic-en-1B (Indic languages to English)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
from functools import lru_cache
from typing import Tuple

logger = logging.getLogger(__name__)

# Reduce likelihood of tokenizer/multiprocessing issues on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")  # Rust threadpool used by tokenizers
os.environ.setdefault("OMP_NUM_THREADS", "1")     # OpenMP threads (torch/tokenizers)
os.environ.setdefault("MKL_NUM_THREADS", "1")     # MKL threads if available
try:
    import multiprocessing as _mp
    if _mp.get_start_method(allow_none=True) != "spawn":
        _mp.set_start_method("spawn", force=True)
except Exception:
    # Safe to ignore if already set or not supported in this context
    pass

# Supported languages with their native names
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'हिन्दी',
    'ta': 'தமிழ்',
    'te': 'తెలుగు',
    'bn': 'বাংলা',
    'mr': 'मराठी',
    'kn': 'ಕನ್ನಡ',
    'ml': 'മലയാളം',
    'gu': 'ગુજરાતી',
    'pa': 'ਪੰਜਾਬੀ',
    'or': 'ଓଡ଼ିଆ',
    'as': 'অসমীয়া'
}

# IndicTrans2 language codes (script-based format)
INDIC_CODES = {
    'en': 'eng_Latn',
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

# Agricultural domain glossary for domain-specific terms (English -> other languages)
AGRICULTURAL_GLOSSARY = {
    # Basic agricultural terms
    'nitrogen': {'hi': 'नाइट्रोजन', 'ta': 'நைட்ரஜன்', 'te': 'నత్రజని', 'kn': 'ನೈಟ್ರೋಜನ್', 'bn': 'নাইট্রোজেন'},
    'phosphorus': {'hi': 'फास्फोरस', 'ta': 'பாஸ்பரஸ்', 'te': 'భాస్వరం', 'kn': 'ರಂಜಕ', 'bn': 'ফসফরাস'},
    'potassium': {'hi': 'पोटैशियम', 'ta': 'பொட்டாசியம்', 'te': 'పొటాషియం', 'kn': 'ಪೊಟ್ಯಾಸಿಯಮ್', 'bn': 'পটাশিয়াম'},
    'fertilizer': {'hi': 'उर्वरक', 'ta': 'உரம்', 'te': 'ఎరువు', 'kn': 'ರಸಗೊಬ್ಬರ', 'bn': 'সার'},
    'soil': {'hi': 'मिट्टी', 'ta': 'மண்', 'te': 'మట్టి', 'kn': 'ಮಣ್ಣು', 'bn': 'মাটি'},
    'crop': {'hi': 'फसल', 'ta': 'பயிர்', 'te': 'పంట', 'kn': 'ಬೆಳೆ', 'bn': 'ফসল'},
    'rainfall': {'hi': 'वर्षा', 'ta': 'மழை', 'te': 'వర్షపాతం', 'kn': 'ಮಳೆ', 'bn': 'বৃষ্টিপাত'},
    'temperature': {'hi': 'तापमान', 'ta': 'வெப்பநிலை', 'te': 'ఉష్ణోగ్రత', 'kn': 'ತಾಪಮಾನ', 'bn': 'তাপমাত্রা'},
    'humidity': {'hi': 'आर्द्रता', 'ta': 'ஈரப்பதம்', 'te': 'తేమ', 'kn': 'ಆರ್ದ್ರತೆ', 'bn': 'আর্দ্রতা'},
    'pH': {'hi': 'पीएच', 'ta': 'pH', 'te': 'pH', 'kn': 'pH', 'bn': 'pH'},
    
    # Common crop names
    'rice': {'hi': 'धान', 'ta': 'நெல்', 'te': 'వరి', 'kn': 'ಅಕ್ಕಿ', 'bn': 'ধান'},
    'wheat': {'hi': 'गेहूं', 'ta': 'கோதுமை', 'te': 'గోధుమ', 'kn': 'ಗೋಧಿ', 'bn': 'গম'},
    'maize': {'hi': 'मक्का', 'ta': 'சோளம்', 'te': 'మొక్కజొన్న', 'kn': 'ಜೋಳ', 'bn': 'ভুট্টা'},
    'corn': {'hi': 'मक्का', 'ta': 'சோளம்', 'te': 'మొక్కజొన్న', 'kn': 'ಜೋಳ', 'bn': 'ভুট্টা'},
    'cotton': {'hi': 'कपास', 'ta': 'பருத்தி', 'te': 'దూది', 'kn': 'ಹತ್ತಿ', 'bn': 'তুলা'},
    'sugarcane': {'hi': 'गन्ना', 'ta': 'கரும்பு', 'te': 'చెరకు', 'kn': 'ಕಬ್ಬು', 'bn': 'আখ'},
    'jute': {'hi': 'जूट', 'ta': 'சணல்', 'te': 'జనపనార', 'kn': 'ಸೆಣಬು', 'bn': 'পাট'},
    'chickpea': {'hi': 'चना', 'ta': 'கொண்டைக்கடலை', 'te': 'శనగలు', 'kn': 'ಕಡಲೆ', 'bn': 'ছোলা'},
    'lentil': {'hi': 'मसूर', 'ta': 'பயிறு', 'te': 'కందులు', 'kn': 'ತುವರೆ', 'bn': 'মসুর'},
    'millet': {'hi': 'बाजरा', 'ta': 'கம்பு', 'te': 'సజ్జలు', 'kn': 'ರಾಗಿ', 'bn': 'বাজরা'},
    'sorghum': {'hi': 'ज्वार', 'ta': 'சோளம்', 'te': 'జొన్న', 'kn': 'ಜೋಳ', 'bn': 'জোয়ার'},
    'groundnut': {'hi': 'मूंगफली', 'ta': 'நிலக்கடலை', 'te': 'వేరుశెనగ', 'kn': 'ಕಡಲೆಕಾಯಿ', 'bn': 'চিনাবাদাম'},
    'soybean': {'hi': 'सोयाबीन', 'ta': 'சோயா பீன்', 'te': 'సోయాబీన్', 'kn': 'ಸೋಯಾ ಬೀಜ', 'bn': 'সয়াবিন'},
    'sunflower': {'hi': 'सूरजमुखी', 'ta': 'சூரியகாந்தி', 'te': 'సూర్యకాంతి', 'kn': 'ಸೂರ್ಯಕಾಂತಿ', 'bn': 'সূর্যমুখী'},
    'potato': {'hi': 'आलू', 'ta': 'உருளைக்கிழங்கு', 'te': 'బంగాళాదుంప', 'kn': 'ಆಲೂಗೆಡ್ಡೆ', 'bn': 'আলু'},
    'tomato': {'hi': 'टमाटर', 'ta': 'தக்காளி', 'te': 'టమోటా', 'kn': 'ಟೊಮ್ಯಾಟೊ', 'bn': 'টমেটো'},
    'onion': {'hi': 'प्याज', 'ta': 'வெங்காயம்', 'te': 'ఉల్లిపాయ', 'kn': 'ಈರುಳ್ಳಿ', 'bn': 'পেঁয়াজ'},
    'banana': {'hi': 'केला', 'ta': 'வாழை', 'te': 'అరటి', 'kn': 'ಬಾಳೆ', 'bn': 'কলা'},
    'mango': {'hi': 'आम', 'ta': 'மாங்காய்', 'te': 'మామిడి', 'kn': 'ಮಾವು', 'bn': 'আম'},
    'apple': {'hi': 'सेब', 'ta': 'ஆப்பிள்', 'te': 'ఆపిల్', 'kn': 'ಸೇಬು', 'bn': 'আপেল'},
    'grape': {'hi': 'अंगूर', 'ta': 'திராட்சை', 'te': 'ద్రాక్ష', 'kn': 'ದ್ರಾಕ್ಷಿ', 'bn': 'আঙুর'},
    'orange': {'hi': 'संतरा', 'ta': 'ஆரஞ்சு', 'te': 'నారింజ', 'kn': 'ಕಿತ್ತಳೆ', 'bn': 'কমলা'},
    'coconut': {'hi': 'नारियल', 'ta': 'தேங்காய்', 'te': 'కొబ్బరి', 'kn': 'ತೆಂಗಿನಕಾಯಿ', 'bn': 'নারকেল'},
    'tea': {'hi': 'चाय', 'ta': 'தேயிலை', 'te': 'తేనీరు', 'kn': 'ಚಹಾ', 'bn': 'চা'},
    'coffee': {'hi': 'कॉफी', 'ta': 'காபி', 'te': 'కాఫీ', 'kn': 'ಕಾಫಿ', 'bn': 'কফি'},
    
    # Extended crop glossary for complete coverage (all 11 Indian languages)
    'barley': {'hi': 'जौ', 'ta': 'வாற்கோதுமை', 'te': 'బార్లీ', 'kn': 'ಬಾರ್ಲಿ', 'bn': 'যব', 'mr': 'जव', 'ml': 'യവം', 'gu': 'જવ', 'pa': 'ਜੌਂ', 'or': 'ଯବ', 'as': 'যৱ'},
    'papaya': {'hi': 'पपीता', 'ta': 'பப்பாளி', 'te': 'బొప్పాయి', 'kn': 'ಪಪ್ಪಾಯ', 'bn': 'পেঁপে', 'mr': 'पपई', 'ml': 'പപ്പായ', 'gu': 'પપૈયા', 'pa': 'ਪਪੀਤਾ', 'or': 'ଅମୃତଭଣ୍ଡା', 'as': 'অমিতা'},
    'watermelon': {'hi': 'तरबूज', 'ta': 'தர்பூசணி', 'te': 'పుచ్చకాయ', 'kn': 'ಕಲ್ಲಂಗಡಿ', 'bn': 'তরমুজ', 'mr': 'टरबूज', 'ml': 'തണ്ണിമത്തൻ', 'gu': 'તરબૂચ', 'pa': 'ਤਰਬੂਜ', 'or': 'ତରଭୁଜ', 'as': 'তৰমুজ'},
    'muskmelon': {'hi': 'खरबूजा', 'ta': 'முலாம்பழம்', 'te': 'ఖర్బూజా', 'kn': 'ಮಸ್ಕ್ಮೆಲನ್', 'bn': 'খরমুজ', 'mr': 'खरबूज', 'ml': 'മധുരപ്പഴം', 'gu': 'ખરબૂજ', 'pa': 'ਖਰਬੂਜਾ', 'or': 'ଖରଭୁଜା', 'as': 'তিয়ঁহ'},
    'pomegranate': {'hi': 'अनार', 'ta': 'மாதுளை', 'te': 'దానిమ్మ', 'kn': 'ದಾಳಿಂಬೆ', 'bn': 'ডালিম', 'mr': 'डाळिंब', 'ml': 'മാതളനാരകം', 'gu': 'દાડમ', 'pa': 'ਅਨਾਰ', 'or': 'ଡାଳିମ୍ବ', 'as': 'ডালিম'},
    'grapes': {'hi': 'अंगूर', 'ta': 'திராட்சை', 'te': 'ద్రాక్ష', 'kn': 'ದ್ರಾಕ್ಷಿ', 'bn': 'আঙুর', 'mr': 'द्राक्षे', 'ml': 'മുന്തിരിങ്ങ', 'gu': 'દ્રાક્ષ', 'pa': 'ਅੰਗੂਰ', 'or': 'ଦ୍ରାକ୍ଷା', 'as': 'আঙুৰ'},
    'blackgram': {'hi': 'उड़द', 'ta': 'உளுந்து', 'te': 'మినుముల', 'kn': 'ಉದ್ದು', 'bn': 'কালো মাষকলাই', 'mr': 'उडीद', 'ml': 'ഉഴുന്ന്', 'gu': 'અડદ', 'pa': 'ਮਾਂਹ', 'or': 'ବିରି', 'as': 'মাটিমাহ'},
    'mungbean': {'hi': 'मूंग', 'ta': 'பயறு', 'te': 'పెసలు', 'kn': 'ಹೆಸರು', 'bn': 'মুগ ডাল', 'mr': 'मूग', 'ml': 'ചെറുപയർ', 'gu': 'મગ', 'pa': 'ਮੂੰਗ', 'or': 'ମୁଗ', 'as': 'মাটিমাহ'},
    'mothbeans': {'hi': 'मोठ', 'ta': 'பருப்பு', 'te': 'మోత్', 'kn': 'ಮೊತ್', 'bn': 'মথ', 'mr': 'मठ', 'ml': 'മോത്ത്', 'gu': 'મોઠ', 'pa': 'ਮੋਠ', 'or': 'ମୋଠ', 'as': 'মাহকলাই'},
    'pigeonpeas': {'hi': 'अरहर', 'ta': 'துவரை', 'te': 'కందులు', 'kn': 'ತೊಗರಿ', 'bn': 'অড়হর', 'mr': 'तूर', 'ml': 'തുവര', 'gu': 'તુવેર', 'pa': 'ਅਰਹਰ', 'or': 'ହରଡ଼', 'as': 'অৰহৰ'},
    'kidneybeans': {'hi': 'राजमा', 'ta': 'ராஜ்மா', 'te': 'రాజ్మా', 'kn': 'ರಾಜಮ', 'bn': 'রাজমা', 'mr': 'राजमा', 'ml': 'രാജ്മ', 'gu': 'રાજમા', 'pa': 'ਰਾਜਮਾ', 'or': 'ରାଜମା', 'as': 'ৰাজমাহ'},
    'chickpeas': {'hi': 'चना', 'ta': 'கொண்டைக்கடலை', 'te': 'శనగలు', 'kn': 'ಕಡಲೆ', 'bn': 'ছোলা', 'mr': 'हरभरा', 'ml': 'കടല', 'gu': 'ચણા', 'pa': 'ਚਣਾ', 'or': 'ଚଣା', 'as': 'বুট'},
    'lentils': {'hi': 'मसूर', 'ta': 'பயிறு', 'te': 'కందులు', 'kn': 'ತುವರೆ', 'bn': 'মসুর', 'mr': 'मसूर', 'ml': 'മസൂർ', 'gu': 'મસૂર', 'pa': 'ਮਸੂਰੀ', 'or': 'ମସୁର', 'as': 'মচুৰ'},
    'mango': {'hi': 'आम', 'ta': 'மாம்பழம்', 'te': 'మామిడి', 'kn': 'ಮಾವು', 'bn': 'আম', 'mr': 'आंबा', 'ml': 'മാങ്ങ', 'gu': 'કેરી', 'pa': 'ਅੰਬ', 'or': 'ଆମ୍ବ', 'as': 'আম'},
    'apple': {'hi': 'सेब', 'ta': 'ஆப்பிள்', 'te': 'ఆపిల్', 'kn': 'ಸೇಬು', 'bn': 'আপেল', 'mr': 'सफरचंद', 'ml': 'ആപ്പിൾ', 'gu': 'સફરજન', 'pa': 'ਸੇਬ', 'or': 'ସେଓ', 'as': 'আপেল'},
    'orange': {'hi': 'संतरा', 'ta': 'ஆரஞ்சு', 'te': 'నారింజ', 'kn': 'ಕಿತ್ತಳೆ', 'bn': 'কমলা', 'mr': 'संत्रा', 'ml': 'ഓറഞ്ച്', 'gu': 'નારંગી', 'pa': 'ਸੰਤਰਾ', 'or': 'କମଳା', 'as': 'কমলা'},
    'papaya': {'hi': 'पपीता', 'ta': 'பப்பாளி', 'te': 'బొప్పాయి', 'kn': 'ಪಪ್ಪಾಯ', 'bn': 'পেঁপে', 'mr': 'पपई', 'ml': 'പപ്പായ', 'gu': 'પપૈયા', 'pa': 'ਪਪੀਤਾ', 'or': 'ଅମୃତଭଣ୍ଡା', 'as': 'অমিতা'},
    'coconut': {'hi': 'नारियल', 'ta': 'தேங்காய்', 'te': 'కొబ్బరి', 'kn': 'ತೆಂಗಿನಕಾಯಿ', 'bn': 'নারকেল', 'mr': 'नारळ', 'ml': 'തേങ്ങ', 'gu': 'નાળિયેર', 'pa': 'ਨਾਰੀਅਲ', 'or': 'ନଡ଼ିଆ', 'as': 'নাৰিকল'},
    'jute': {'hi': 'जूट', 'ta': 'சணல்', 'te': 'జనపనార', 'kn': 'ಸೆಣಬು', 'bn': 'পাট', 'mr': 'ताग', 'ml': 'ചണം', 'gu': 'શણ', 'pa': 'ਪਟਸਣ', 'or': 'ପଟ', 'as': 'মেস্তা'},
    'coffee': {'hi': 'कॉफी', 'ta': 'காபி', 'te': 'కాఫీ', 'kn': 'ಕಾಫಿ', 'bn': 'কফি', 'mr': 'कॉफी', 'ml': 'കാപ്പി', 'gu': 'કોફી', 'pa': 'ਕੌਫੀ', 'or': 'କଫି', 'as': 'কফি'}
}


class TranslationCache:
    """Simple in-memory cache for translations."""
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, str] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[str]:
        """Get cached translation."""
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, value: str):
        """Cache translation."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'size': len(self._cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%"
        }


class TranslationService:
    """Translation service using AI4Bharat's IndicTrans2 models."""
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize translation service with IndicTrans2.
        
        Args:
            use_cache: Whether to use caching for translations
        """
        self.use_cache = use_cache
        self.cache = TranslationCache() if use_cache else None
        
        # Lazy loading - models loaded on first use
        self.models_loaded = False
        self.ip = None  # IndicProcessor
        self.en_indic_tokenizer = None
        self.en_indic_model = None
        self.indic_en_tokenizer = None
        self.indic_en_model = None
        
        # Load static translations from JSON file
        self.static_translations = self._load_static_translations()
        
        logger.info(f"Translation service initialized with IndicTrans2. Cache: {use_cache}")
    
    def _load_static_translations(self) -> Dict:
        """Load pre-translated static content from JSON file."""
        translations_file = Path(__file__).parent.parent.parent / 'translations' / 'messages.json'
        
        if translations_file.exists():
            try:
                with open(translations_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load translations file: {e}")
                return {}
        else:
            logger.warning(f"Translations file not found: {translations_file}")
            return {}

    def _get_hf_token(self) -> Optional[str]:
        """Retrieve Hugging Face token from environment if provided."""
        # Prefer HF_TOKEN, but also support HUGGINGFACE_HUB_TOKEN
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        return token if token and token.strip() else None
    
    def _load_indictrans2_models(self):
        """Load IndicTrans2 models (lazy loading on first use)."""
        if self.models_loaded:
            return
        
        # Emergency bypass: skip model load if explicitly disabled
        if os.getenv("SKIP_INDICTRANS2_MODELS", "0") == "1":
            logger.warning("IndicTrans2 models SKIPPED (SKIP_INDICTRANS2_MODELS=1). Dynamic translation disabled.")
            self.models_loaded = False
            return
        
        try:
            logger.info("Loading IndicTrans2 models... This may take a minute on first run.")
            
            from IndicTransToolkit import IndicProcessor
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import transformers as _tf
            from packaging import version as _pkg_version
            from huggingface_hub import login as hf_login
            
            # Model checkpoints (optionally pin a revision via env)
            en_indic_ckpt = os.getenv("INDICTRANS2_EN_INDIC", "ai4bharat/indictrans2-en-indic-1B")
            indic_en_ckpt = os.getenv("INDICTRANS2_INDIC_EN", "ai4bharat/indictrans2-indic-en-1B")
            pinned_rev = os.getenv("INDICTRANS2_REVISION")  # optional commit/tag
            
            # Initialize processor
            self.ip = IndicProcessor(inference=True)
            
            # Resolve HF auth token if available
            hf_token = self._get_hf_token()
            if hf_token:
                try:
                    # Log in so gated repos and rate limits are handled
                    hf_login(token=hf_token)
                    logger.info("Authenticated with Hugging Face Hub using provided HF_TOKEN.")
                except Exception as e:
                    logger.warning(f"Hugging Face login failed; will pass token directly to loaders. Error: {e}")

            # Build kwargs for model/tokenizer auth compatible across versions
            def _auth_kwargs() -> Dict:
                kw: Dict = {}
                if not hf_token:
                    return kw
                try:
                    # Newer transformers support 'token'
                    _ = AutoTokenizer.from_pretrained
                    # We'll test support dynamically below in calls
                    kw["token"] = hf_token
                except Exception:
                    kw["use_auth_token"] = hf_token
                return kw

            auth_kw = _auth_kwargs()

            # Load English to Indic model
            logger.info(f"Loading {en_indic_ckpt}...")
            try:
                self.en_indic_tokenizer = AutoTokenizer.from_pretrained(
                    en_indic_ckpt,
                    trust_remote_code=True,
                    revision=pinned_rev if pinned_rev else None,
                    **auth_kw
                )
            except TypeError:
                # Fallback for older API
                legacy_kw = {"use_auth_token": hf_token} if hf_token else {}
                self.en_indic_tokenizer = AutoTokenizer.from_pretrained(
                    en_indic_ckpt,
                    trust_remote_code=True,
                    revision=pinned_rev if pinned_rev else None,
                    **legacy_kw
                )
            try:
                self.en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                    en_indic_ckpt,
                    trust_remote_code=True,
                    revision=pinned_rev if pinned_rev else None,
                    **auth_kw
                )
            except TypeError:
                legacy_kw = {"use_auth_token": hf_token} if hf_token else {}
                self.en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                    en_indic_ckpt,
                    trust_remote_code=True,
                    revision=pinned_rev if pinned_rev else None,
                    **legacy_kw
                )
            
            # Load Indic to English model
            logger.info(f"Loading {indic_en_ckpt}...")
            try:
                self.indic_en_tokenizer = AutoTokenizer.from_pretrained(
                    indic_en_ckpt,
                    trust_remote_code=True,
                    revision=pinned_rev if pinned_rev else None,
                    **auth_kw
                )
            except TypeError:
                legacy_kw = {"use_auth_token": hf_token} if hf_token else {}
                self.indic_en_tokenizer = AutoTokenizer.from_pretrained(
                    indic_en_ckpt,
                    trust_remote_code=True,
                    revision=pinned_rev if pinned_rev else None,
                    **legacy_kw
                )
            try:
                self.indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
                    indic_en_ckpt,
                    trust_remote_code=True,
                    revision=pinned_rev if pinned_rev else None,
                    **auth_kw
                )
            except TypeError:
                legacy_kw = {"use_auth_token": hf_token} if hf_token else {}
                self.indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
                    indic_en_ckpt,
                    trust_remote_code=True,
                    revision=pinned_rev if pinned_rev else None,
                    **legacy_kw
                )
            
            self.models_loaded = True
            logger.info("IndicTrans2 models loaded successfully!")
            
        except ImportError as e:
            logger.error(f"IndicTrans2 not installed. Install with: pip install IndicTransToolkit transformers sentencepiece sacremoses")
            logger.error(f"Error: {e}")
            self.models_loaded = False
        except Exception as e:
            logger.error(f"Failed to load IndicTrans2 models: {e}")
            self.models_loaded = False
    
    def _get_indic_code(self, lang: str) -> str:
        """Convert 2-letter language code to IndicTrans2 format."""
        return INDIC_CODES.get(lang, 'eng_Latn')
    
    def _check_glossary(self, text: str, target_lang: str) -> Optional[str]:
        """Check if text matches agricultural glossary term."""
        text_lower = text.lower().strip()
        if text_lower in AGRICULTURAL_GLOSSARY:
            if target_lang in AGRICULTURAL_GLOSSARY[text_lower]:
                return AGRICULTURAL_GLOSSARY[text_lower][target_lang]
        return None
    
    def _translate_with_indictrans2(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text using IndicTrans2 models.
        
        Args:
            text: Text to translate
            source_lang: Source language code (2-letter)
            target_lang: Target language code (2-letter)
        
        Returns:
            Translated text or None if failed
        """
        if not self.models_loaded:
            self._load_indictrans2_models()
        
        if not self.models_loaded:
            return None
        
        try:
            # Convert to IndicTrans2 format
            src_code = self._get_indic_code(source_lang)
            tgt_code = self._get_indic_code(target_lang)
            
            # Determine which model to use
            if source_lang == 'en' and target_lang != 'en':
                # English to Indic
                tokenizer = self.en_indic_tokenizer
                model = self.en_indic_model
            elif source_lang != 'en' and target_lang == 'en':
                # Indic to English
                tokenizer = self.indic_en_tokenizer
                model = self.indic_en_model
            else:
                # Indic to Indic (pivot through English)
                logger.debug(f"Pivot translation: {source_lang} -> en -> {target_lang}")
                intermediate = self._translate_with_indictrans2(text, source_lang, 'en')
                if intermediate:
                    return self._translate_with_indictrans2(intermediate, 'en', target_lang)
                return None
            
            # Preprocess
            batch = self.ip.preprocess_batch(
                [text],
                src_lang=src_code,
                tgt_lang=tgt_code
            )
            
            # Tokenize
            inputs = tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True
            )
            
            # Generate translation
            generated_tokens = model.generate(
                **inputs,
                num_beams=5,
                num_return_sequences=1,
                max_length=256
            )
            
            # Decode
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Postprocess
            translations = self.ip.postprocess_batch(
                generated_tokens,
                lang=tgt_code
            )
            
            translated = translations[0] if translations else None
            if translated:
                logger.debug(f"Translated: '{text[:30]}...' -> '{translated[:30]}...'")
            
            return translated
            
        except Exception as e:
            logger.error(f"IndicTrans2 translation error: {e}")
            return None
    
    def get_static_translation(self, key: str, lang: str, default: Optional[str] = None) -> str:
        """
        Get pre-translated static content.
        
        Args:
            key: Translation key (e.g., 'ui.crop_recommendation')
            lang: Target language code
            default: Default text if translation not found
        
        Returns:
            Translated text or default
        """
        try:
            # Try to get from static translations for the target language
            if lang in self.static_translations and key in self.static_translations[lang]:
                return self.static_translations[lang][key]
            
            # Fallback to English if target language not found
            if 'en' in self.static_translations and key in self.static_translations['en']:
                return self.static_translations['en'][key]
        except Exception as e:
            logger.error(f"Error getting static translation for {key} in {lang}: {e}")
        
        # Return default or key as last resort
        return default if default is not None else key
    
    def translate_text(self, text: str, source_lang: str = 'en', target_lang: str = 'hi') -> str:
        """
        Translate text using IndicTrans2.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            Translated text or original if translation fails
        """
        # No translation needed
        if source_lang == target_lang or not text or not text.strip():
            return text
        
        # Check agricultural glossary first
        glossary_match = self._check_glossary(text, target_lang)
        if glossary_match:
            logger.debug(f"Glossary match: {text} -> {glossary_match}")
            return glossary_match
        
        # Check cache
        if self.cache:
            cache_key = f"{source_lang}:{target_lang}:{text}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for: {text[:50]}...")
                return cached
        
        # Try IndicTrans2 translation
        try:
            translated = self._translate_with_indictrans2(text, source_lang, target_lang)
            if translated:
                # Cache the result
                if self.cache:
                    self.cache.set(cache_key, translated)
                return translated
        except Exception as e:
            logger.error(f"Translation error: {e}")
        
        # Fallback: return original text
        logger.warning(f"Translation failed for '{text[:50]}...', returning original")
        return text
    
    def translate_batch(self, texts: List[str], source_lang: str = 'en', target_lang: str = 'hi') -> List[str]:
        """
        Translate multiple texts efficiently.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            List of translated texts
        """
        if source_lang == target_lang:
            return texts
        
        results = []
        for text in texts:
            translated = self.translate_text(text, source_lang, target_lang)
            results.append(translated)
        
        return results
    
    def translate_dict(self, data: Dict, keys_to_translate: List[str], 
                      source_lang: str = 'en', target_lang: str = 'hi') -> Dict:
        """
        Translate specific keys in a dictionary.
        
        Args:
            data: Dictionary containing text to translate
            keys_to_translate: List of keys whose values should be translated
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            Dictionary with translated values
        """
        if source_lang == target_lang:
            return data
        
        result = data.copy()
        for key in keys_to_translate:
            if key in result and isinstance(result[key], str):
                result[key] = self.translate_text(result[key], source_lang, target_lang)
        
        return result
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {'cache_enabled': False}
    
    def clear_cache(self):
        """Clear translation cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Translation cache cleared")


# Global translation service instance
_translation_service: Optional[TranslationService] = None


def get_translation_service() -> TranslationService:
    """Get or create global translation service instance."""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service


def translate(text: str, target_lang: str = 'en', source_lang: str = 'en') -> str:
    """
    Convenience function to translate text.
    
    Args:
        text: Text to translate
        target_lang: Target language code
        source_lang: Source language code
    
    Returns:
        Translated text
    """
    service = get_translation_service()
    return service.translate_text(text, source_lang, target_lang)


def t(key: str, lang: str = 'en', **kwargs) -> str:
    """
    Get translated text from static translations with optional formatting.
    
    Args:
        key: Translation key
        lang: Language code
        **kwargs: Format arguments for string interpolation
    
    Returns:
        Translated and formatted text
    """
    service = get_translation_service()
    # Fetch target-language static value (may be missing or placeholder)
    text = service.get_static_translation(key, lang, default=None)
    # Also fetch the English source text for dynamic fallback
    en_text = service.get_static_translation(key, 'en', default=key)

    # If non-English and static looks missing/English, try dynamic translation
    if lang != 'en':
        def _looks_english(s: Optional[str]) -> bool:
            if not s or not isinstance(s, str):
                return True
            has_letter = any(ch.isalpha() for ch in s)
            is_ascii_only = all(ord(ch) < 128 for ch in s)
            return has_letter and is_ascii_only

        needs_dynamic = (
            text is None or
            text == key or
            (en_text and text is not None and text.strip() == en_text.strip()) or
            _looks_english(text)
        )

        if needs_dynamic and en_text:
            try:
                translated = service.translate_text(en_text, source_lang='en', target_lang=lang)
                if translated and isinstance(translated, str) and translated.strip():
                    text = translated
                else:
                    # Fallback to whatever static we had or English
                    text = text if text else en_text
            except Exception as e:
                logger.warning(f"Dynamic fallback translation failed for key '{key}' to '{lang}': {e}")
                text = text if text else en_text

    # Final fallback: if still missing, use English
    if text is None:
        text = en_text if en_text is not None else key
    
    # Apply string formatting if kwargs provided
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception as e:
            logger.error(f"Error formatting translation '{key}': {e}")
    
    return text


def get_supported_languages() -> Dict:
    """Get dictionary of supported languages."""
    return SUPPORTED_LANGUAGES.copy()


def is_language_supported(lang_code: str) -> bool:
    """Check if language code is supported."""
    return lang_code in SUPPORTED_LANGUAGES
