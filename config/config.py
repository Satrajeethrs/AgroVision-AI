"""
Configuration management for AgroVision-AI
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

class Config:
    """Base configuration"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'True') == 'True'
    
    # Application settings
    APP_NAME = os.getenv('APP_NAME', 'AgroVision-AI')
    APP_VERSION = os.getenv('APP_VERSION', '1.0.0')
    PORT = int(os.getenv('PORT', 5001))
    
    # LLM Configuration
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'none')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
    ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229')
    
    # Model paths
    MODEL_PATH = BASE_DIR / os.getenv('MODEL_PATH', 'models/model.pkl')
    SCALER_PATH = BASE_DIR / os.getenv('SCALER_PATH', 'models/scaler.pkl')
    DISEASE_MODEL_PATH = BASE_DIR / os.getenv('DISEASE_MODEL_PATH', 'models/plant_disease_model_final.h5')
    CLASS_INDICES_PATH = BASE_DIR / os.getenv('CLASS_INDICES_PATH', 'models/plant_disease_class_indices.npy')
    
    # Data paths
    CROP_DATA_PATH = BASE_DIR / os.getenv('CROP_DATA_PATH', 'data/raw/crop_data.csv')
    FERTILIZER_DATA_PATH = BASE_DIR / os.getenv('FERTILIZER_DATA_PATH', 'data/raw/fertilizer.csv')
    
    # Upload configuration
    MAX_UPLOAD_SIZE = int(os.getenv('MAX_UPLOAD_SIZE', 5242880))  # 5MB
    ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'jpg,jpeg,png').split(','))
    
    # Validation ranges
    VALIDATION_RANGES = {
        'N': {'min': 0, 'max': 140, 'unit': 'kg/ha'},
        'P': {'min': 5, 'max': 145, 'unit': 'kg/ha'},
        'K': {'min': 5, 'max': 205, 'unit': 'kg/ha'},
        'temperature': {'min': 8, 'max': 44, 'unit': '°C'},
        'humidity': {'min': 14, 'max': 100, 'unit': '%'},
        'ph': {'min': 3.5, 'max': 9.9, 'unit': ''},
        'rainfall': {'min': 20, 'max': 300, 'unit': 'mm'}
    }
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = BASE_DIR / os.getenv('LOG_FILE', 'logs/app.log')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    # Use test databases/models for testing

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(env=None):
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])
