"""
Script to add chatbot translation keys to messages.json
"""

import json
from pathlib import Path

# Path to translations file
translations_file = Path(__file__).parent / 'translations' / 'messages.json'

# New chatbot translation keys
chatbot_keys = {
    "en": {
        # Chatbot page
        "chatbot.title": "AI Farming Assistant",
        "chatbot.subtitle": "Ask questions and get expert agricultural advice in your language",
        "chatbot.configuration": "Configuration",
        "chatbot.provider": "AI Provider",
        "chatbot.auto_detect": "Auto-detect",
        "chatbot.provider_help": "Choose your AI provider or use auto-detect",
        "chatbot.status": "Status",
        "chatbot.initializing": "Initializing...",
        "chatbot.ready": "Ready to chat",
        "chatbot.start": "Start Chatting",
        "chatbot.connected": "Connected",
        "chatbot.conversation": "Conversation",
        "chatbot.type_message": "Type your farming question here...",
        "chatbot.send": "Send",
        "chatbot.export": "Export",
        "chatbot.clear": "Clear",
        "chatbot.you": "You",
        "chatbot.bot": "AI Assistant",
        "chatbot.thinking": "Thinking...",
        "chatbot.error": "Sorry, I encountered an error. Please try again.",
        "chatbot.cleared": "Chat history cleared. Start a new conversation!",
        "chatbot.welcome": "Hello! I'm your AI farming assistant. Ask me anything about crops, fertilizers, pests, diseases, or farming practices.",
        "chatbot.tips": "Tip: You can ask about crop selection, fertilizer use, disease management, and more!",
        "chatbot.sample_questions": "Sample Questions",
        "chatbot.about": "About AI Farming Assistant",
        "chatbot.about_text": "This intelligent chatbot uses advanced AI to provide personalized farming advice. It understands your language and context to give practical, actionable recommendations.",
        "chatbot.feature1": "Multi-language support for all Indian languages",
        "chatbot.feature2": "Context-aware responses based on your farm analysis",
        "chatbot.feature3": "Expert knowledge on crops, soil, pests, and diseases",
        "chatbot.feature4": "Practical, cost-effective farming recommendations",
        "chatbot.confirm_clear": "Are you sure you want to clear the chat history?",
        
        # Navigation
        "nav.ai_chatbot": "AI Chatbot Assistant",
        "nav.back_to_home": "Back to Home"
    }
}

def update_translations():
    """Add chatbot keys to all languages in messages.json"""
    
    # Load existing translations
    with open(translations_file, 'r', encoding='utf-8') as f:
        translations = json.load(f)
    
    # Add English keys
    if 'en' in translations:
        translations['en'].update(chatbot_keys['en'])
        print("✓ Added English chatbot translations")
    
    # For other languages, add keys with placeholder text that will be dynamically translated
    for lang_code in translations.keys():
        if lang_code != 'en' and lang_code != '_comment':
            for key, value in chatbot_keys['en'].items():
                if key not in translations[lang_code]:
                    # Add the English text as placeholder (will be translated dynamically)
                    translations[lang_code][key] = value
            print(f"✓ Added placeholder chatbot translations for {lang_code}")
    
    # Save updated translations
    with open(translations_file, 'w', encoding='utf-8') as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Successfully updated {translations_file}")
    print("Note: Non-English translations will be handled dynamically by IndicTrans2")

if __name__ == '__main__':
    update_translations()
