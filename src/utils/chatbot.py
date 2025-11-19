"""
Chatbot utilities for AgroVision-AI.

This module provides an AI-powered chatbot for farmers with support for:
- Multiple LLM providers (Gemini, LM Studio, OpenAI)
- Context-aware responses based on crop data
- Multilingual support through translation service
- Farmer-specific system prompts
- Session-based conversation history
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat messages and return response."""
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.api_key = api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GEMINI_KEY')
        self.base_url = os.getenv('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com')
        self.model = 'gemini-pro'
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat to Gemini API."""
        if not self.api_key:
            raise ValueError("Gemini API key not found")
        
        try:
            # Convert messages to Gemini format
            contents = []
            for msg in messages:
                role = 'user' if msg['role'] == 'user' else 'model'
                contents.append({
                    'role': role,
                    'parts': [{'text': msg['content']}]
                })
            
            url = f"{self.base_url}/v1/models/{self.model}:generateContent?key={self.api_key}"
            
            payload = {
                'contents': contents,
                'generationConfig': {
                    'temperature': kwargs.get('temperature', 0.7),
                    'maxOutputTokens': kwargs.get('max_tokens', 1024),
                }
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if 'candidates' in data and len(data['candidates']) > 0:
                return data['candidates'][0]['content']['parts'][0]['text']
            
            return "I'm sorry, I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


class LMStudioProvider(LLMProvider):
    """LM Studio local API provider."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key, base_url)
        self.base_url = base_url or os.getenv('LMSTUDIO_BASE_URL', 'http://127.0.0.1:1234')
        # LM Studio uses OpenAI-compatible API
        self.api_key = api_key or os.getenv('LMSTUDIO_API_KEY') or os.getenv('LM_STUDIO_KEY') or 'not-needed'
        self.model = os.getenv('LMSTUDIO_MODEL')  # optional, will auto-detect if missing

        # If running inside Docker and base_url is 127.0.0.1, rewrite to host.docker.internal
        try:
            if os.path.exists('/.dockerenv') and '127.0.0.1' in self.base_url:
                self.base_url = self.base_url.replace('127.0.0.1', 'host.docker.internal')
                logger.info(f"LM Studio base URL rewritten for Docker: {self.base_url}")
        except Exception:
            pass

        # Attempt early model auto-detection if not set
        if not self.model:
            try:
                m = self._early_pick_model(self.base_url)
                if m:
                    self.model = m
                    logger.info(f"LM Studio model selected: {self.model}")
            except Exception as e:
                logger.debug(f"Early model detection skipped: {e}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat to LM Studio API."""
        def _call(base: str, with_model: Optional[str]) -> str:
            url = f"{base}/v1/chat/completions"
            
            # LM Studio models often only support user/assistant roles
            # Convert system messages to user messages
            cleaned_messages = []
            system_prompt = None
            for msg in messages:
                if msg['role'] == 'system':
                    system_prompt = msg['content']
                else:
                    cleaned_messages.append(msg)
            
            # Prepend system prompt to first user message if exists
            if system_prompt and cleaned_messages:
                for i, msg in enumerate(cleaned_messages):
                    if msg['role'] == 'user':
                        cleaned_messages[i] = {
                            'role': 'user',
                            'content': f"{system_prompt}\n\nUser question: {msg['content']}"
                        }
                        break
            elif system_prompt:
                # No user message yet, add as user
                cleaned_messages.insert(0, {'role': 'user', 'content': system_prompt})
            
            payload = {
                'messages': cleaned_messages,
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 1024),
                'stream': False
            }
            if with_model:
                payload['model'] = with_model
            headers = {'Content-Type': 'application/json'}
            if self.api_key and self.api_key != 'not-needed':
                headers['Authorization'] = f'Bearer {self.api_key}'
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0]['message']['content']
            return "I'm sorry, I couldn't generate a response."

        def _pick_model(base: str) -> Optional[str]:
            try:
                models_url = f"{base}/v1/models"
                headers = {'Content-Type': 'application/json'}
                if self.api_key and self.api_key != 'not-needed':
                    headers['Authorization'] = f'Bearer {self.api_key}'
                r = requests.get(models_url, headers=headers, timeout=10)
                r.raise_for_status()
                data = r.json()
                # Prefer chat/instruct models first
                if isinstance(data, dict) and 'data' in data and data['data']:
                    items = data['data']
                    def _mid(i):
                        return i.get('id') or i.get('name') or ''
                    preferred = [i for i in items if any(k in _mid(i).lower() for k in ['instruct', 'chat', 'qwen', 'llama', 'mistral'])]
                    chosen = preferred[0] if preferred else items[0]
                    model_id = _mid(chosen)
                    logger.info(f"LM Studio available model detected: {model_id}")
                    return model_id
            except Exception as e:
                logger.warning(f"Failed to auto-detect LM Studio model: {e}")
            return None

        # helper used in __init__
        self._early_pick_model = _pick_model

        # Try primary URL, then a Docker-friendly fallback if connection refused
        try:
            return _call(self.base_url, self.model)
        except requests.HTTPError as he:
            # If 400 (Bad Request), try to auto-pick model and retry
            status = getattr(he.response, 'status_code', None)
            text = None
            try:
                text = he.response.text
            except Exception:
                pass
            logger.warning(f"LM Studio HTTPError {status}: {text}")

            if status == 400:
                # Try with detected model on same base URL
                autodetected = _pick_model(self.base_url)
                if autodetected:
                    try:
                        self.model = autodetected
                        return _call(self.base_url, self.model)
                    except Exception as e2:
                        logger.warning(f"Retry with autodetected model failed: {e2}")

            # Fallback path to host.docker.internal
            msg = str(he)
            if '127.0.0.1' in self.base_url:
                fallback = self.base_url.replace('127.0.0.1', 'host.docker.internal')
                autodetected = _pick_model(fallback)
                try:
                    logger.info(f"Retrying LM Studio via fallback: {fallback} with model: {autodetected or self.model}")
                    out = _call(fallback, autodetected or self.model)
                    self.base_url = fallback
                    if autodetected:
                        self.model = autodetected
                    return out
                except Exception as e2:
                    logger.error(f"LM Studio fallback failed: {e2}")
            raise
        except requests.Timeout as te:
            logger.error(f"LM Studio timeout after 120s - model may be too slow or not loaded: {te}")
            raise Exception("LM Studio response timed out. The model may be too large or not fully loaded. Try a smaller/faster model.")
        except Exception as e:
            msg = str(e)
            logger.warning(f"LM Studio call failed on {self.base_url}: {msg}")
            if '127.0.0.1' in self.base_url:
                fallback = self.base_url.replace('127.0.0.1', 'host.docker.internal')
                try:
                    logger.info(f"Retrying LM Studio via fallback: {fallback}")
                    out = _call(fallback, self.model)
                    # Update base_url for subsequent calls
                    self.base_url = fallback
                    return out
                except Exception as e2:
                    logger.error(f"LM Studio fallback failed: {e2}")
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key, base_url)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com')
        self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat to OpenAI API."""
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        try:
            url = f"{self.base_url}/v1/chat/completions"
            
            payload = {
                'model': self.model,
                'messages': messages,
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 1024),
            }
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0]['message']['content']
            
            return "I'm sorry, I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


def get_farmer_system_prompt(language: str = 'en', user_context: Optional[Dict] = None) -> str:
    """
    Generate a specialized system prompt for farmer assistance.
    
    Args:
        language: User's preferred language
        user_context: Optional context about user's farm/crops
    
    Returns:
        System prompt string
    """
    base_prompt = """You are an expert agricultural advisor and agronomist specializing in helping farmers make data-driven decisions. Your role is to:

1. Provide practical, actionable agricultural advice
2. Explain complex farming concepts in simple, easy-to-understand terms
3. Consider local climate, soil conditions, and seasonal factors
4. Recommend sustainable and economically viable farming practices
5. Answer questions about crop selection, fertilizer use, pest management, irrigation, and soil health
6. Help interpret soil test results and environmental data
7. Suggest disease prevention and treatment strategies
8. Provide guidance on crop rotation and intercropping

Important Guidelines:
- Keep answers concise but informative (3-5 sentences typically)
- Use simple language that farmers can easily understand
- Provide specific, actionable recommendations
- Consider cost-effectiveness and practicality
- When discussing chemicals or treatments, always mention safety precautions
- If you're unsure, suggest consulting with local agricultural extension services
- Encourage sustainable and organic farming practices when appropriate

"""
    
    # Add user context if available
    if user_context:
        if 'crop' in user_context:
            base_prompt += f"\nThe farmer is currently growing or interested in: {user_context['crop']}\n"
        if 'soil_type' in user_context:
            base_prompt += f"Their soil type is: {user_context['soil_type']}\n"
        if 'location' in user_context:
            base_prompt += f"Farm location: {user_context['location']}\n"
    
    # Add language instruction
    if language != 'en':
        lang_names = {
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'bn': 'Bengali',
            'mr': 'Marathi',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'gu': 'Gujarati',
            'pa': 'Punjabi',
            'or': 'Odia',
            'as': 'Assamese'
        }
        lang_name = lang_names.get(language, language)
        base_prompt += f"\nIMPORTANT: Respond in {lang_name} language. The farmer speaks {lang_name}.\n"
    
    return base_prompt


class FarmerChatbot:
    """AI chatbot specifically designed for farmer assistance."""
    
    def __init__(self, provider_type: str = 'auto', language: str = 'en'):
        """
        Initialize the chatbot.
        
        Args:
            provider_type: 'gemini', 'lmstudio', 'openai', or 'auto'
            language: User's preferred language code
        """
        self.language = language
        self.provider = self._initialize_provider(provider_type)
        self.conversation_history: List[Dict[str, str]] = []
        self.user_context: Dict[str, Any] = {}
        
        logger.info(f"Chatbot initialized with provider: {type(self.provider).__name__}")
    
    def _initialize_provider(self, provider_type: str) -> LLMProvider:
        """Initialize the appropriate LLM provider."""
        
        if provider_type == 'auto':
            # Auto-detect based on available credentials
            if os.getenv('GEMINI_API_KEY') or os.getenv('GEMINI_KEY'):
                provider_type = 'gemini'
            elif os.getenv('LMSTUDIO_BASE_URL'):
                provider_type = 'lmstudio'
            elif os.getenv('OPENAI_API_KEY'):
                provider_type = 'openai'
            else:
                # Default to LM Studio (local)
                provider_type = 'lmstudio'
        
        providers = {
            'gemini': GeminiProvider,
            'lmstudio': LMStudioProvider,
            'openai': OpenAIProvider
        }
        
        if provider_type not in providers:
            raise ValueError(f"Unknown provider: {provider_type}")
        
        return providers[provider_type]()
    
    def set_context(self, context: Dict[str, Any]):
        """
        Set user context (farm details, current analysis, etc.).
        
        Args:
            context: Dictionary with user/farm context
        """
        self.user_context.update(context)
        logger.info(f"Updated user context: {list(context.keys())}")
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def chat(self, user_message: str, include_context: bool = True) -> Dict[str, Any]:
        """
        Send a message and get response.
        
        Args:
            user_message: User's message
            include_context: Whether to include system prompt with context
        
        Returns:
            Dict with response and metadata
        """
        try:
            # Build messages list
            messages = []
            
            # Add system prompt if first message or context included
            if len(self.conversation_history) == 0 or include_context:
                system_prompt = get_farmer_system_prompt(
                    language=self.language,
                    user_context=self.user_context
                )
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            
            # Add conversation history (last 5 exchanges to avoid token limits)
            history_limit = 10  # 5 exchanges = 10 messages
            recent_history = self.conversation_history[-history_limit:] if len(self.conversation_history) > history_limit else self.conversation_history
            messages.extend(recent_history)
            
            # Add current user message
            messages.append({
                'role': 'user',
                'content': user_message
            })
            
            # Get response from provider
            response = self.provider.chat(messages, temperature=0.7, max_tokens=1024)
            
            # Update conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': user_message
            })
            self.conversation_history.append({
                'role': 'assistant',
                'content': response
            })
            
            return {
                'success': True,
                'response': response,
                'provider': type(self.provider).__name__.replace('Provider', ''),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': "I'm sorry, I encountered an error. Please try again or check your LLM provider configuration.",
                'timestamp': datetime.now().isoformat()
            }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    def export_conversation(self) -> str:
        """Export conversation as JSON string."""
        return json.dumps({
            'language': self.language,
            'context': self.user_context,
            'history': self.conversation_history,
            'timestamp': datetime.now().isoformat()
        }, indent=2)


def create_chatbot(language: str = 'en', provider: str = 'auto') -> FarmerChatbot:
    """
    Factory function to create a chatbot instance.
    
    Args:
        language: User's preferred language
        provider: LLM provider to use
    
    Returns:
        FarmerChatbot instance
    """
    return FarmerChatbot(provider_type=provider, language=language)
