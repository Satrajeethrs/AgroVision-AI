# 🌾 AgroVision-AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An intelligent agricultural advisory system powered by Machine Learning and Deep Learning to help farmers make data-driven decisions about crop selection, fertilizer management, and plant disease detection.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Model Information](#model-information)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

AgroVision-AI is a comprehensive agricultural decision support system that combines multiple AI models to provide:

- **Crop Recommendation**: ML-based recommendations based on soil nutrients (N, P, K), climate conditions (temperature, humidity, rainfall), and soil pH
- **Fertilizer Optimization**: Smart fertilizer recommendations with detailed application guidelines
- **Plant Disease Detection**: Deep learning-based visual disease identification from leaf images
- **LLM-Enhanced Insights**: AI-generated narratives and alternative recommendations

## ✨ Features

### 🌱 Smart Crop Recommendation
- Analyzes 7 key agricultural parameters
- Uses Random Forest classification with 95%+ accuracy
- Provides confidence scores and alternative suggestions
- Considers soil nutrients, climate, and pH levels

### 🧪 Intelligent Fertilizer Management
- Personalized NPK (Nitrogen, Phosphorus, Potassium) recommendations
- Application timing and dosage guidelines
- Crop-specific fertilizer schedules
- Soil health management advice

### 🔍 Plant Disease Detection
- Deep CNN model trained on 38+ disease classes
- Real-time image-based disease identification
- Treatment recommendations and severity assessment
- Supports multiple crop types (Tomato, Potato, Pepper, etc.)

### 📊 Comprehensive Analytics
- Detailed environmental condition analysis
- Soil nutrient status visualization
- Executive summaries and actionable insights
- pH and moisture management recommendations

## 📁 Project Structure

```
AgroVision-AI/
├── app.py                      # Main Flask application
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── core/                   # Core functionality
│   │   ├── __init__.py
│   │   ├── train_model.py                    # Crop recommendation model training
│   │   ├── train_plant_disease_model.py      # Disease detection model training
│   │   └── prepare_plant_disease_dataset.py  # Dataset preparation
│   │
│   └── utils/                  # Utility modules
│       ├── __init__.py
│       ├── input_validation.py  # Input validation logic
│       ├── formatters.py        # Output formatting utilities
│       └── llm_validator.py     # LLM integration for insights
│
├── models/                     # Trained models
│   ├── model.pkl               # Crop recommendation model
│   ├── scaler.pkl              # Feature scaler
│   ├── plant_disease_model_final.h5      # Disease detection model
│   ├── plant_disease_model_best.h5       # Best checkpoint
│   └── plant_disease_class_indices.npy   # Class mappings
│
├── data/                       # Data files
│   ├── raw/                    # Raw datasets
│   │   ├── crop_data.csv       # Crop recommendation dataset
│   │   └── fertilizer.csv      # Fertilizer recommendations
│   └── processed/              # Processed data
│
├── templates/                  # HTML templates
│   ├── index.html              # Main input form
│   └── results.html            # Results display
│
├── static/                     # Static assets
│   ├── style.css               # Main styles
│   ├── results.css             # Results page styles
│   ├── validation.js           # Client-side validation
│   └── results.js              # Results page interactions
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_validator.py
│   ├── test_llm_narrative.py
│   ├── test_integration_ai_narrative.py
│   ├── test_validate_endpoint.py
│   ├── comprehensive_test.py
│   ├── test_plant_disease_model.py
│   └── validate_recs.py
│
├── archive/                    # Archived datasets
│   └── New Plant Diseases Dataset(Augmented)/
│
├── docs/                       # Documentation
├── config/                     # Configuration files
└── .env                        # Environment variables
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Satrajeeth/AgroVision-AI.git
cd AgroVision-AI
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

```bash
# Create .env file (optional, for LLM features)
cp .env.example .env

# Edit .env and add your API keys if using LLM features
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
```

## 🏃 Quick Start

### Option 1: Use Pre-trained Models (Recommended)

If the models are already trained and available in the `models/` directory:

```bash
python app.py
```

Then open your browser and navigate to: `http://localhost:5001`

### Option 2: Train Models from Scratch

#### Train Crop Recommendation Model

```bash
python src/core/train_model.py
```

This will:
- Load the crop dataset from `data/raw/crop_data.csv`
- Perform hyperparameter tuning
- Train a Random Forest classifier
- Save the model to `models/model.pkl` and scaler to `models/scaler.pkl`

#### Train Plant Disease Detection Model

```bash
python src/core/train_plant_disease_model.py
```

This will:
- Load images from `archive/New Plant Diseases Dataset(Augmented)/`
- Build and train a CNN model
- Save the best model to `models/plant_disease_model_best.h5`
- Save class indices to `models/plant_disease_class_indices.npy`

### Start the Web Application

```bash
python app.py
```

The application will be available at: `http://localhost:5001`

## 📖 Usage Guide

### Web Interface

1. **Enter Soil Parameters**:
   - Nitrogen (N): 0-140 kg/ha
   - Phosphorus (P): 5-145 kg/ha
   - Potassium (K): 5-205 kg/ha
   - Soil pH: 3.5-9.9

2. **Enter Climate Data**:
   - Temperature: 8-44°C
   - Humidity: 14-100%
   - Rainfall: 20-300 mm

3. **Optional: Upload Plant Image**:
   - Upload a clear image of plant leaves
   - Supported formats: JPG, PNG
   - Get instant disease detection results

4. **View Results**:
   - Executive summary with recommendations
   - Detailed crop suitability analysis
   - Fertilizer recommendations with application guidelines
   - Disease detection results (if image uploaded)
   - AI-generated insights and alternatives

### API Endpoints

#### POST /analyze
Submit agricultural data and optional plant image for analysis.

**Request Body (form-data)**:
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.87,
  "humidity": 82,
  "ph": 6.5,
  "rainfall": 202.9,
  "disease_image": "<file>"
}
```

**Response**: Redirects to `/results`

#### GET /results
View the analysis results (requires session data from `/analyze`)

#### POST /validate_recs
Validate recommendations using LLM (requires session data)

**Response**:
```json
{
  "validation": {
    "status": "success",
    "recommendations": [...]
  }
}
```

## 🧠 Model Information

### Crop Recommendation Model

- **Algorithm**: Random Forest Classifier
- **Features**: N, P, K, temperature, humidity, pH, rainfall (7 features)
- **Classes**: 22 crop types
- **Accuracy**: ~95-97% on test set
- **Training**: GridSearchCV with 5-fold cross-validation

**Feature Importance**:
1. Rainfall (highest)
2. Humidity
3. Potassium (K)
4. Phosphorus (P)
5. Nitrogen (N)
6. Temperature
7. pH

### Plant Disease Detection Model

- **Architecture**: Custom CNN
- **Input Size**: 128×128 RGB images
- **Classes**: 38 disease categories
- **Layers**:
  - 3 Convolutional blocks (32, 64, 128 filters)
  - MaxPooling after each block
  - Dense layer (256 units) + Dropout (0.5)
  - Softmax output layer
- **Optimizer**: Adam (lr=0.0001)
- **Training**: 40 epochs with data augmentation
- **Accuracy**: ~90-95% on validation set

**Supported Diseases**:
- Apple: Apple scab, Black rot, Cedar apple rust
- Tomato: Bacterial spot, Early blight, Late blight, Leaf mold, etc.
- Potato: Early blight, Late blight
- Pepper: Bacterial spot
- And more...

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# LLM Provider (optional)
LLM_PROVIDER=openai  # or anthropic, or none
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Model Paths
MODEL_PATH=models/model.pkl
SCALER_PATH=models/scaler.pkl
DISEASE_MODEL_PATH=models/plant_disease_model_final.h5
```

### Input Validation Ranges

The system enforces the following validation ranges:

| Parameter   | Min   | Max   | Unit   |
|------------|-------|-------|--------|
| Nitrogen   | 0     | 140   | kg/ha  |
| Phosphorus | 5     | 145   | kg/ha  |
| Potassium  | 5     | 205   | kg/ha  |
| Temperature| 8     | 44    | °C     |
| Humidity   | 14    | 100   | %      |
| pH         | 3.5   | 9.9   | -      |
| Rainfall   | 20    | 300   | mm     |

## 🧪 Testing

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Suites

```bash
# Test validators
pytest tests/test_validator.py

# Test LLM integration
pytest tests/test_llm_narrative.py

# Test plant disease model
pytest tests/test_plant_disease_model.py

# Integration tests
pytest tests/test_integration_ai_narrative.py
```

### Test Coverage

```bash
pytest --cov=src tests/
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit: `git commit -am 'Add feature'`
6. Push: `git push origin feature-name`
7. Submit a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Write docstrings for functions and classes
- Add tests for new features

## 📊 Performance Benchmarks

| Component | Metric | Value |
|-----------|--------|-------|
| Crop Model | Accuracy | 95-97% |
| Disease Model | Accuracy | 90-95% |
| API Response | Time | <500ms |
| Image Processing | Time | <2s |

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'src'`
```bash
# Solution: Ensure you're in the project root directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Issue**: Models not loading
```bash
# Solution: Train the models first
python src/core/train_model.py
python src/core/train_plant_disease_model.py
```

**Issue**: TensorFlow errors on Apple Silicon
```bash
# Solution: Install TensorFlow for macOS
pip install tensorflow-macos tensorflow-metal
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Plant disease dataset from [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- Crop recommendation dataset from agricultural research databases
- Flask and TensorFlow communities for excellent documentation

## 📞 Contact

- **Author**: Satrajeeth
- **GitHub**: [@Satrajeeth](https://github.com/Satrajeeth)
- **Project**: [AgroVision-AI](https://github.com/Satrajeeth/AgroVision-AI)

## 🌐 Multilingual Support

AgroVision-AI now supports **12 Indian languages** powered by AI4Bharat's Bhashini API:

- 🇮🇳 **English** (en) - Default
- 🇮🇳 **हिन्दी** (hi) - Hindi
- 🇮🇳 **தமிழ்** (ta) - Tamil
- 🇮🇳 **తెలుగు** (te) - Telugu
- 🇮🇳 **বাংলা** (bn) - Bengali
- 🇮🇳 **मराठी** (mr) - Marathi
- 🇮🇳 **ಕನ್ನಡ** (kn) - Kannada
- 🇮🇳 **മലയാളം** (ml) - Malayalam
- 🇮🇳 **ગુજરાતી** (gu) - Gujarati
- 🇮🇳 **ਪੰਜਾਬੀ** (pa) - Punjabi
- 🇮🇳 **ଓଡ଼ିଆ** (or) - Odia
- 🇮🇳 **অসমীয়া** (as) - Assamese

### Features
- **Language Selector**: Available in header on every page
- **Static Translations**: UI labels, buttons, and forms
- **Dynamic Translation**: Recommendations and advice
- **LLM Multilingual**: AI-generated content in user's language

### Setup
Add to your `.env` file:
```bash
BHASHINI_API_KEY=your_api_key_here
ENABLE_TRANSLATION=True
```

See detailed documentation: [Multilingual Support Guide](docs/MULTILINGUAL.md)

## 🗺️ Roadmap

- [x] Multi-language support (12 Indian languages)
- [ ] Mobile application (iOS/Android)
- [ ] Weather API integration
- [ ] Soil testing API integration
- [ ] Marketplace for agricultural products
- [ ] Community forum for farmers
- [ ] Real-time crop monitoring dashboard
- [ ] Voice input/output in local languages

---

**Made with ❤️ for farmers and agriculture enthusiasts**
