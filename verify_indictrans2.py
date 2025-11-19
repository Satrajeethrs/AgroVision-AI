"""
Quick verification test for IndicTrans2 integration.
This test checks that the translation service can be initialized and configured properly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from src.utils.translation import (
            get_translation_service,
            translate,
            t,
            get_supported_languages,
            is_language_supported,
            SUPPORTED_LANGUAGES,
            INDIC_CODES,
            AGRICULTURAL_GLOSSARY
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test that translation service is configured correctly."""
    print("\nTesting configuration...")
    try:
        from src.utils.translation import SUPPORTED_LANGUAGES, INDIC_CODES, AGRICULTURAL_GLOSSARY
        
        # Check languages
        assert len(SUPPORTED_LANGUAGES) == 12, f"Expected 12 languages, got {len(SUPPORTED_LANGUAGES)}"
        assert 'en' in SUPPORTED_LANGUAGES, "English not in supported languages"
        assert 'hi' in SUPPORTED_LANGUAGES, "Hindi not in supported languages"
        
        # Check IndicTrans2 codes
        assert len(INDIC_CODES) == 12, f"Expected 12 language codes, got {len(INDIC_CODES)}"
        assert INDIC_CODES['en'] == 'eng_Latn', "English code incorrect"
        assert INDIC_CODES['hi'] == 'hin_Deva', "Hindi code incorrect"
        
        # Check glossary
        assert 'nitrogen' in AGRICULTURAL_GLOSSARY, "nitrogen not in glossary"
        assert 'hi' in AGRICULTURAL_GLOSSARY['nitrogen'], "Hindi translation missing for nitrogen"
        
        print("✅ Configuration correct")
        return True
    except AssertionError as e:
        print(f"❌ Configuration error: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_service_initialization():
    """Test that translation service can be initialized."""
    print("\nTesting service initialization...")
    try:
        from src.utils.translation import get_translation_service
        
        service = get_translation_service()
        assert service is not None, "Service is None"
        assert hasattr(service, 'translate_text'), "translate_text method missing"
        assert hasattr(service, 'translate_batch'), "translate_batch method missing"
        assert hasattr(service, 'get_static_translation'), "get_static_translation method missing"
        
        print("✅ Service initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False

def test_helper_functions():
    """Test helper functions."""
    print("\nTesting helper functions...")
    try:
        from src.utils.translation import get_supported_languages, is_language_supported
        
        languages = get_supported_languages()
        assert len(languages) == 12, f"Expected 12 languages, got {len(languages)}"
        
        assert is_language_supported('en') == True, "English not recognized"
        assert is_language_supported('hi') == True, "Hindi not recognized"
        assert is_language_supported('xx') == False, "Invalid language recognized"
        
        print("✅ Helper functions working")
        return True
    except Exception as e:
        print(f"❌ Helper function test failed: {e}")
        return False

def test_static_translations():
    """Test static translation loading."""
    print("\nTesting static translations...")
    try:
        from src.utils.translation import get_translation_service
        
        service = get_translation_service()
        
        # Try to get a static translation (may not exist, that's ok)
        result = service.get_static_translation('ui.welcome', 'en', default='Welcome')
        assert result is not None, "Static translation returned None"
        
        print("✅ Static translation system working")
        return True
    except Exception as e:
        print(f"❌ Static translation test failed: {e}")
        return False

def test_cache():
    """Test translation cache."""
    print("\nTesting cache...")
    try:
        from src.utils.translation import get_translation_service
        
        service = get_translation_service()
        
        # Check cache exists
        assert service.cache is not None, "Cache is None"
        
        # Test cache operations
        service.cache.set('test:key', 'test value')
        result = service.cache.get('test:key')
        assert result == 'test value', "Cache get/set failed"
        
        # Get stats
        stats = service.get_cache_stats()
        assert 'size' in stats, "Cache stats missing 'size'"
        assert 'hits' in stats, "Cache stats missing 'hits'"
        
        print("✅ Cache working correctly")
        return True
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False

def test_glossary_check():
    """Test agricultural glossary lookup."""
    print("\nTesting glossary lookup...")
    try:
        from src.utils.translation import get_translation_service
        
        service = get_translation_service()
        
        # Test glossary lookup
        result = service._check_glossary('nitrogen', 'hi')
        assert result is not None, "Glossary lookup failed for 'nitrogen'"
        assert result == 'नाइट्रोजन', f"Wrong translation: expected 'नाइट्रोजन', got '{result}'"
        
        # Test non-existent term
        result = service._check_glossary('nonexistent', 'hi')
        assert result is None, "Non-existent term returned value"
        
        print("✅ Glossary lookup working")
        return True
    except Exception as e:
        print(f"❌ Glossary test failed: {e}")
        return False

def test_no_translation_needed():
    """Test that same-language translation returns original."""
    print("\nTesting same-language translation...")
    try:
        from src.utils.translation import translate
        
        text = "Hello World"
        result = translate(text, target_lang='en', source_lang='en')
        assert result == text, "Same language translation didn't return original"
        
        print("✅ Same-language translation working")
        return True
    except Exception as e:
        print(f"❌ Same-language test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("="*60)
    print("IndicTrans2 Integration Verification")
    print("="*60)
    
    tests = [
        test_imports,
        test_configuration,
        test_service_initialization,
        test_helper_functions,
        test_static_translations,
        test_cache,
        test_glossary_check,
        test_no_translation_needed,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        print("\nNext steps:")
        print("1. Install IndicTrans2 dependencies:")
        print("   pip install IndicTransToolkit transformers sentencepiece sacremoses torch")
        print("\n2. On first translation, models will download (~4GB, one-time)")
        print("\n3. Test with actual translation:")
        print("   from src.utils.translation import translate")
        print("   print(translate('Hello', target_lang='hi', source_lang='en'))")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please review the errors above and fix before proceeding.")
        return 1

if __name__ == '__main__':
    exit(main())
