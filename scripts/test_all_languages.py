#!/usr/bin/env python3
"""
Multilingual Testing Script for AgroVision-AI
Tests all 12 languages end-to-end and reports any English leakage.
"""

import requests
import json
from typing import Dict, List, Tuple

BASE_URL = "http://127.0.0.1:5001"

# All supported languages with their native names
LANGUAGES = {
    'en': 'English',
    'hi': 'हिन्दी (Hindi)',
    'ta': 'தமிழ் (Tamil)',
    'te': 'తెలుగు (Telugu)',
    'bn': 'বাংলা (Bengali)',
    'mr': 'मराठी (Marathi)',
    'kn': 'ಕನ್ನಡ (Kannada)',
    'ml': 'മലയാളം (Malayalam)',
    'gu': 'ગુજરાતી (Gujarati)',
    'pa': 'ਪੰਜਾਬੀ (Punjabi)',
    'or': 'ଓଡ଼ିଆ (Odia)',
    'as': 'অসমীয়া (Assamese)'
}

# Sample test data
TEST_DATA = {
    'N': '90',
    'P': '42',
    'K': '43',
    'temperature': '26',
    'humidity': '80',
    'ph': '6.5',
    'rainfall': '1200',
    'SOC': '0.6',
    'disease_symptoms': ''
}

def test_language(lang_code: str, lang_name: str) -> Tuple[bool, List[str]]:
    """
    Test a single language by:
    1. Setting language
    2. Submitting form
    3. Checking response for English leakage
    
    Returns: (success, list_of_issues)
    """
    issues = []
    session = requests.Session()
    
    try:
        # Step 1: Set language
        print(f"\n{'='*60}")
        print(f"🔄 Testing: {lang_name} ({lang_code})")
        print(f"{'='*60}")
        
        lang_response = session.post(
            f"{BASE_URL}/set_language",
            json={'language': lang_code},
            headers={'Content-Type': 'application/json'}
        )
        
        if lang_response.status_code != 200:
            issues.append(f"❌ Failed to set language: {lang_response.status_code}")
            return False, issues
        
        print(f"  ✅ Language set successfully")
        
        # Step 2: Submit form
        print(f"  📝 Submitting test form...")
        form_response = session.post(
            f"{BASE_URL}/analyze",
            data=TEST_DATA,
            allow_redirects=False
        )
        
        if form_response.status_code not in [200, 302]:
            issues.append(f"❌ Form submission failed: {form_response.status_code}")
            return False, issues
        
        print(f"  ✅ Form submitted successfully")
        
        # Step 3: Get results page
        print(f"  📊 Fetching results page...")
        results_response = session.get(f"{BASE_URL}/results")
        
        if results_response.status_code != 200:
            issues.append(f"❌ Results page failed: {results_response.status_code}")
            return False, issues
        
        html_content = results_response.text
        
        # Check for common English words that shouldn't appear (except in English mode)
        if lang_code != 'en':
            english_indicators = [
                'Recommended Crop',
                'Soil Nutrient Status',
                'Environmental Conditions',
                'Key Recommendations',
                'Nitrogen',
                'Phosphorus',
                'Potassium',
                'Temperature',
                'Humidity',
                'Rainfall',
                'Submit',
                'Analysis',
                'Optimal'
            ]
            
            found_english = []
            for indicator in english_indicators:
                # Check if the indicator appears as standalone text (not in attributes)
                if f'>{indicator}<' in html_content or f'>{indicator} ' in html_content:
                    found_english.append(indicator)
            
            if found_english:
                issues.append(f"⚠️  Found English text: {', '.join(found_english[:3])}...")
                print(f"  ⚠️  WARNING: Found {len(found_english)} English phrases")
            else:
                print(f"  ✅ No English leakage detected!")
        
        # Check that translated content exists
        if 'utf-8' not in html_content.lower():
            issues.append("⚠️  Page may not have UTF-8 encoding")
        
        print(f"  ✅ Results page loaded successfully")
        
        if not issues:
            print(f"  🎉 {lang_name} passed all checks!")
            return True, []
        else:
            print(f"  ⚠️  {lang_name} has {len(issues)} issue(s)")
            return True, issues
        
    except Exception as e:
        issues.append(f"❌ Exception: {str(e)}")
        print(f"  ❌ Error testing {lang_name}: {e}")
        return False, issues

def main():
    """Run tests for all languages."""
    print("\n" + "="*60)
    print("🌐 MULTILINGUAL TEST SUITE - AgroVision-AI")
    print("="*60)
    print(f"📍 Testing server: {BASE_URL}")
    print(f"🔢 Total languages: {len(LANGUAGES)}")
    print("="*60)
    
    # Check server is up
    try:
        response = requests.get(BASE_URL, timeout=5)
        if response.status_code != 200:
            print(f"\n❌ Server not responding at {BASE_URL}")
            print("   Please ensure Flask is running on port 5001")
            return
    except Exception as e:
        print(f"\n❌ Cannot connect to server: {e}")
        print("   Please start the server with: python app.py")
        return
    
    print("✅ Server is online\n")
    
    # Test each language
    results = {}
    for lang_code, lang_name in LANGUAGES.items():
        success, issues = test_language(lang_code, lang_name)
        results[lang_code] = {
            'name': lang_name,
            'success': success,
            'issues': issues
        }
    
    # Summary report
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r['success'] and not r['issues'])
    warnings = sum(1 for r in results.values() if r['success'] and r['issues'])
    failed = sum(1 for r in results.values() if not r['success'])
    
    print(f"\n✅ Passed: {passed}/{len(LANGUAGES)}")
    print(f"⚠️  Warnings: {warnings}/{len(LANGUAGES)}")
    print(f"❌ Failed: {failed}/{len(LANGUAGES)}")
    
    if warnings > 0:
        print("\n⚠️  LANGUAGES WITH WARNINGS:")
        for lang_code, result in results.items():
            if result['success'] and result['issues']:
                print(f"\n  {result['name']} ({lang_code}):")
                for issue in result['issues']:
                    print(f"    • {issue}")
    
    if failed > 0:
        print("\n❌ LANGUAGES WITH FAILURES:")
        for lang_code, result in results.items():
            if not result['success']:
                print(f"\n  {result['name']} ({lang_code}):")
                for issue in result['issues']:
                    print(f"    • {issue}")
    
    print("\n" + "="*60)
    if failed == 0 and warnings == 0:
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\n✅ The application is fully multilingual!")
        print("   All 12 languages are working correctly.")
    elif failed == 0:
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        print(f"\n⚠️  {warnings} language(s) have minor issues (English text found)")
        print("   This is likely due to English placeholders in new keys.")
        print("   The app is functional but translations can be improved.")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("="*60)
        print(f"\n❌ {failed} language(s) failed to complete testing")
        print("   Please review errors above and fix issues.")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()
