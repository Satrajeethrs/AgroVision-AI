# 🌾 AgroVision-AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)

> An intelligent agricultural advisory system powered by Machine Learning and Deep Learning to help farmers make data-driven decisions about crop selection, fertilizer management, and plant disease detection.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Model Information](#-model-information)
- [Multilingual Support](#-multilingual-support)
- [Docker Deployment](#-docker-deployment)
- [Development](#-development)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

## 🌟 Overview

AgroVision-AI is a comprehensive agricultural decision support system that combines multiple AI models to provide:

- **🌱 Crop Recommendation**: ML-based recommendations using soil nutrients (N, P, K), climate conditions (temperature, humidity, rainfall), and soil pH
- **🧪 Fertilizer Optimization**: Smart fertilizer recommendations with detailed application guidelines
- **🔍 Plant Disease Detection**: Deep learning-based visual disease identification from leaf images
- **💬 AI Chatbot**: Interactive agricultural assistant powered by LLMs (OpenAI/Gemini)
- **🌐 Multilingual Support**: 12 Indian languages powered by AI4Bharat's IndicTrans2

## ✨ Features

### Smart Crop Recommendation
- Analyzes 7 key agricultural parameters
- Uses Random Forest classification with 95%+ accuracy
- Provides confidence scores and alternative suggestions
- Considers soil nutrients, climate, and pH levels

### Intelligent Fertilizer Management
- Personalized NPK (Nitrogen, Phosphorus, Potassium) recommendations
- Application timing and dosage guidelines
- Crop-specific fertilizer schedules
- Soil health management advice

### Plant Disease Detection
- Deep CNN model trained on 38+ disease classes
- Real-time image-based disease identification
- Treatment recommendations and severity assessment
- Supports multiple crop types (Tomato, Potato, Pepper, Apple, etc.)

### AI-Powered Chatbot
- Context-aware agricultural assistant
- Supports OpenAI GPT and Google Gemini models
- Provides personalized advice based on your analysis
- Multilingual conversation support

### Comprehensive Analytics
- Detailed environmental condition analysis
- Soil nutrient status visualization
- Executive summaries and actionable insights
- pH and moisture management recommendations

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum
- (Optional) Docker for containerized deployment

### 1. Clone Repository
```bash
git clone https://github.com/Satrajeeth/AgroVision-AI.git
cd AgroVision-AI
```

### 2. Quick Setup with Script
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Run Application
```bash
python app.py
```

Visit `http://localhost:5001` in your browser.

## 📦 Installation

### Option 1: Standard Installation

#### Step 1: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

#### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 3: Configure Environment (Optional)
```bash
# Create .env file for LLM features
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here
```

#### Step 4: Train Models (If Not Available)
```bash
# Train crop recommendation model
python src/core/train_model.py

# Train plant disease detection model
python src/core/train_plant_disease_model.py
```

#### Step 5: Run Application
```bash
python app.py
```

### Option 2: Docker Installation

```bash
# Build and run with Docker Compose
docker compose up -d --build

# Access at http://localhost:5001
```

## 📖 Usage

### Web Interface

1. **Navigate to Home Page**: Open `http://localhost:5001`

2. **Enter Soil Parameters**:
   - Nitrogen (N): 0-140 kg/ha
   - Phosphorus (P): 5-145 kg/ha
   - Potassium (K): 5-205 kg/ha
   - Soil pH: 3.5-9.9

3. **Enter Climate Data**:
   - Temperature: 8-44°C
   - Humidity: 14-100%
   - Rainfall: 20-300 mm

4. **Optional - Upload Plant Image**:
   - Upload a clear image of plant leaves
   - Supported formats: JPG, PNG
   - Get instant disease detection results

5. **View Results**:
   - Executive summary with recommendations
   - Detailed crop suitability analysis
   - Fertilizer recommendations with application guidelines
   - Disease detection results (if image uploaded)
   - AI-generated insights and alternatives

6. **Chat with AI Assistant**:
   - Click chatbot icon for interactive help
   - Ask questions about your results
   - Get personalized agricultural advice

### API Usage

#### Analyze Endpoint
```bash
curl -X POST http://localhost:5001/analyze \
  -F "N=90" \
  -F "P=42" \
  -F "K=43" \
  -F "temperature=20.87" \
  -F "humidity=82" \
  -F "ph=6.5" \
  -F "rainfall=202.9" \
  -F "disease_image=@/path/to/leaf.jpg"
```

See [API Documentation](#-api-documentation) for complete details.

## 📁 Project Structure

```
AgroVision-AI/
├── app.py                      # Main Flask application
├── requirements.txt            # Project dependencies
├── setup.sh                    # Automated setup script
├── start_app.sh               # Application startup script
├── Dockerfile                  # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
├── .env.example               # Environment variables template
├── README.md                  # This file
├── LICENSE                    # MIT License
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── core/                  # Core functionality
│   │   ├── __init__.py
│   │   ├── train_model.py                    # Crop model training
│   │   ├── train_plant_disease_model.py      # Disease model training
│   │   └── prepare_plant_disease_dataset.py  # Dataset preparation
│   │
│   └── utils/                 # Utility modules
│       ├── __init__.py
│       ├── input_validation.py    # Input validation
│       ├── formatters.py          # Output formatting
│       ├── llm_validator.py       # LLM integration
│       ├── translation.py         # Multilingual support
│       └── chatbot.py             # AI chatbot
│
├── models/                    # Trained models (gitignored)
│   ├── model.pkl              # Crop recommendation model
│   ├── scaler.pkl             # Feature scaler
│   ├── plant_disease_model_final.h5      # Disease model
│   └── plant_disease_class_indices.npy   # Class mappings
│
├── data/                      # Data files
│   ├── raw/
│   │   ├── crop_data.csv      # Crop dataset
│   │   └── fertilizer.csv     # Fertilizer data
│   └── processed/             # Processed data
│
├── templates/                 # HTML templates
│   ├── index.html            # Main input form
│   ├── results.html          # Results display
│   └── chatbot.html          # Chatbot interface
│
├── static/                    # Static assets
│   ├── style.css             # Main styles
│   ├── results.css           # Results page styles
│   ├── chatbot.css           # Chatbot styles
│   ├── validation.js         # Client-side validation
│   ├── results.js            # Results interactions
│   └── chatbot.js            # Chatbot functionality
│
├── tests/                     # Test suite
│   ├── conftest.py
│   ├── test_validator.py
│   ├── test_llm_narrative.py
│   ├── test_plant_disease_model.py
│   └── comprehensive_test.py
│
├── translations/              # Multilingual support
│   ├── messages.json         # Translation database
│   └── missing_by_lang.json  # Translation tracking
│
├── scripts/                   # Utility scripts
│   ├── audit_translations.py
│   ├── backfill_translations.py
│   ├── propagate_translations.py
│   └── test_all_languages.py
│
├── docs/                      # Documentation
│   ├── API.md                # API documentation
│   ├── CONTRIBUTING.md       # Contribution guidelines
│   ├── DOCKER_SETUP.md       # Docker setup guide
│   └── MULTILINGUAL.md       # Multilingual guide
│
├── config/                    # Configuration
│   └── config.py             # App configuration
│
└── archive/                   # Archived datasets
    └── New Plant Diseases Dataset(Augmented)/
```

## 📡 API Documentation

### Endpoints

#### `POST /analyze`
Submit agricultural data for analysis

**Parameters**:
- `N` (float): Nitrogen content (0-140 kg/ha)
- `P` (float): Phosphorus content (5-145 kg/ha)
- `K` (float): Potassium content (5-205 kg/ha)
- `temperature` (float): Temperature (8-44°C)
- `humidity` (float): Humidity (14-100%)
- `ph` (float): Soil pH (3.5-9.9)
- `rainfall` (float): Rainfall (20-300 mm)
- `disease_image` (file, optional): Plant leaf image

**Response**: Redirects to `/results`

#### `GET /results`
View analysis results

**Response**: HTML page with comprehensive recommendations

#### `POST /validate_recs`
Validate recommendations using LLM

**Response**: JSON with validation results

#### `GET /chatbot`
AI chatbot interface

**Response**: Interactive chat page

#### `POST /api/chatbot/chat`
Send message to chatbot

**Request**:
```json
{
  "message": "How should I apply fertilizer?"
}
```

**Response**:
```json
{
  "status": "success",
  "response": "AI response here...",
  "provider": "openai|gemini"
}
```

#### `POST /set_language`
Change interface language

**Request**:
```json
{
  "language": "hi"
}
```

**Response**:
```json
{
  "status": "success",
  "language": "hi"
}
```

For complete API documentation, see [docs/API.md](docs/API.md).

## 🧠 Model Information

### Crop Recommendation Model

**Algorithm**: Random Forest Classifier  
**Features**: N, P, K, temperature, humidity, pH, rainfall (7 features)  
**Classes**: 22 crop types  
**Accuracy**: ~95-97% on test set  
**Training**: GridSearchCV with 5-fold cross-validation

**Supported Crops**:
- Rice, Maize, Wheat, Cotton, Jute
- Chickpea, Kidney Beans, Pigeon Peas, Moth Beans
- Mung Bean, Black Gram, Lentil
- Pomegranate, Banana, Mango, Grapes, Watermelon
- Muskmelon, Apple, Orange, Papaya, Coconut, Coffee

### Plant Disease Detection Model

**Architecture**: Custom CNN  
**Input Size**: 128×128 RGB images  
**Classes**: 38 disease categories  
**Accuracy**: ~90-95% on validation set

**Layers**:
- 3 Convolutional blocks (32, 64, 128 filters)
- MaxPooling after each block
- Dense layer (256 units) + Dropout (0.5)
- Softmax output layer

**Supported Diseases**:
- **Apple**: Scab, Black rot, Cedar apple rust
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, etc.
- **Potato**: Early blight, Late blight
- **Pepper**: Bacterial spot
- And more (38+ total)

## 🌐 Multilingual Support

AgroVision-AI supports **12 Indian languages** powered by AI4Bharat's IndicTrans2:

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
- Language selector in header
- Static UI translations
- Dynamic content translation
- LLM multilingual responses
- Chatbot support in all languages

### Configuration
Add to `.env`:
```bash
ENABLE_TRANSLATION=True
SKIP_INDICTRANS2_MODELS=0  # Set to 1 on macOS
```

For detailed multilingual documentation, see [docs/MULTILINGUAL.md](docs/MULTILINGUAL.md).

## 🐳 Docker Deployment

### Build and Run

```bash
# Build and start containers
docker compose up -d --build

# View logs
docker compose logs -f

# Stop containers
docker compose down
```

### Docker Configuration

The `docker-compose.yml` configures:
- Flask app on port 5001
- Volume mounts for models and data
- Environment variable loading
- Automatic restart policy

For detailed Docker setup, see [docs/DOCKER_SETUP.md](docs/DOCKER_SETUP.md).

## 💻 Development

### Environment Setup

```bash
# Clone repository
git clone https://github.com/Satrajeeth/AgroVision-AI.git
cd AgroVision-AI

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

### Running in Development Mode

```bash
# Enable debug mode
export FLASK_DEBUG=1

# Run application
python app.py
```

### Code Style

We follow **PEP 8** guidelines:
- 4 spaces indentation
- 100 character line limit
- Type hints for functions
- Google-style docstrings

### Pre-commit Checks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🧪 Testing

### Run All Tests

```bash
pytest tests/
```

### Run Specific Tests

```bash
# Test validators
pytest tests/test_validator.py

# Test LLM integration
pytest tests/test_llm_narrative.py

# Test plant disease model
pytest tests/test_plant_disease_model.py -v
```

### Test with Coverage

```bash
pytest --cov=src tests/
```

### Test Coverage Report

```bash
pytest --cov=src --cov-report=html tests/
# Open htmlcov/index.html
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Quick Start

1. **Fork** the repository
2. **Clone** your fork
3. **Create** a feature branch
4. **Make** your changes
5. **Test** your changes
6. **Commit** with clear messages
7. **Push** to your fork
8. **Submit** a Pull Request

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
pytest tests/

# Commit changes
git commit -m "feat: add your feature"

# Push to fork
git push origin feature/your-feature-name
```

### Commit Message Format

```
<type>(<scope>): <subject>

<body>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Example**:
```
feat(crop): add support for wheat recommendation

Implement wheat-specific logic including optimal NPK ranges
and climate requirements for better accuracy.

Closes #123
```

For detailed contribution guidelines, see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

## 🐛 Troubleshooting

### Common Issues

#### Module Not Found
```bash
# Solution: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Models Not Loading
```bash
# Solution: Train models first
python src/core/train_model.py
python src/core/train_plant_disease_model.py
```

#### Port Already in Use
```bash
# Solution: Kill process or change port
lsof -ti:5001 | xargs kill -9
# Or edit app.py to use different port
```

#### TensorFlow Errors on Apple Silicon
```bash
# Solution: Install TensorFlow for macOS
pip install tensorflow-macos tensorflow-metal
```

#### Translation Errors on macOS
```bash
# Solution: Skip IndicTrans2 models
# Add to .env:
SKIP_INDICTRANS2_MODELS=1
```

### Getting Help

- 📖 Check [documentation](docs/)
- 🐛 Open an [issue](https://github.com/Satrajeeth/AgroVision-AI/issues)
- 💬 Join [discussions](https://github.com/Satrajeeth/AgroVision-AI/discussions)

## 📊 Performance Benchmarks

| Component | Metric | Value |
|-----------|--------|-------|
| Crop Model | Accuracy | 95-97% |
| Disease Model | Accuracy | 90-95% |
| API Response | Time | <500ms |
| Image Processing | Time | <2s |
| Chatbot Response | Time | 1-3s |

## 🗺️ Roadmap

- [x] Multi-language support (12 Indian languages)
- [x] AI Chatbot integration
- [x] Docker deployment
- [ ] Mobile application (iOS/Android)
- [ ] Weather API integration
- [ ] Soil testing API integration
- [ ] Marketplace for agricultural products
- [ ] Community forum for farmers
- [ ] Real-time crop monitoring dashboard
- [ ] Voice input/output in local languages
- [ ] Offline mode support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Plant Disease Dataset**: [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Crop Data**: Agricultural research databases
- **Translation**: AI4Bharat's IndicTrans2
- **Icons**: Open source icon libraries
- **Community**: Contributors and users

## 📞 Contact

- **Author**: Satrajeeth
- **GitHub**: [@Satrajeeth](https://github.com/Satrajeeth)
- **Repository**: [AgroVision-AI](https://github.com/Satrajeeth/AgroVision-AI)
- **Issues**: [Report Bug](https://github.com/Satrajeeth/AgroVision-AI/issues)
- **Discussions**: [Join Discussion](https://github.com/Satrajeeth/AgroVision-AI/discussions)

## 🌟 Star History

If you find this project helpful, please consider giving it a star ⭐️ on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=Satrajeeth/AgroVision-AI&type=Date)](https://star-history.com/#Satrajeeth/AgroVision-AI&Date)

---

**Made with ❤️ for farmers and agriculture enthusiasts**

*Empowering agriculture through artificial intelligence*
