# 🚀 Quick Reference Guide

## Project Structure

```
AgroVision-AI/
├── app.py                 # Main Flask application
├── setup.sh              # Automated setup script
├── requirements.txt      # Python dependencies
├── README.md            # Comprehensive documentation
├── LICENSE              # MIT License
│
├── src/                 # Source code
│   ├── core/           # Model training scripts
│   └── utils/          # Utility functions
│
├── models/             # Trained ML models (gitignored)
├── data/               # Datasets
│   └── raw/           # Raw data files
│
├── templates/          # HTML templates
├── static/            # CSS, JS assets
├── tests/             # Test suite
├── docs/              # Documentation
│   ├── API.md
│   └── CONTRIBUTING.md
│
└── config/            # Configuration files
    └── config.py
```

## Quick Commands

### Setup
```bash
# Automated setup
./setup.sh

# Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training Models
```bash
# Crop recommendation model
python src/core/train_model.py

# Plant disease detection model
python src/core/train_plant_disease_model.py
```

### Running Application
```bash
python app.py
# Opens at http://localhost:5001
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_validator.py -v
```

### Development
```bash
# Activate environment
source venv/bin/activate

# Deactivate environment
deactivate

# Check code style
flake8 src/

# Format code
black src/
```

## File Locations

| Component | Location |
|-----------|----------|
| Main app | `app.py` |
| Models | `models/*.pkl`, `models/*.h5` |
| Data | `data/raw/*.csv` |
| Training | `src/core/train_*.py` |
| Utils | `src/utils/*.py` |
| Tests | `tests/test_*.py` |
| Config | `config/config.py`, `.env` |

## Input Ranges

| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| Nitrogen | 0 | 140 | kg/ha |
| Phosphorus | 5 | 145 | kg/ha |
| Potassium | 5 | 205 | kg/ha |
| Temperature | 8 | 44 | °C |
| Humidity | 14 | 100 | % |
| pH | 3.5 | 9.9 | - |
| Rainfall | 20 | 300 | mm |

## Key Dependencies

- Flask 2.0+ (Web framework)
- TensorFlow 2.0+ (Deep learning)
- scikit-learn (ML models)
- pandas (Data processing)
- numpy (Numerical computing)

## Environment Variables

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Key variables:
- `FLASK_ENV` - development/production
- `SECRET_KEY` - Flask secret key
- `LLM_PROVIDER` - openai/anthropic/none
- `OPENAI_API_KEY` - OpenAI API key (optional)

## Common Issues

**Issue**: Module not found
```bash
# Solution: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Issue**: Models not loading
```bash
# Solution: Train models first
python src/core/train_model.py
python src/core/train_plant_disease_model.py
```

**Issue**: Port already in use
```bash
# Solution: Change port in app.py or kill process
lsof -ti:5001 | xargs kill -9
```

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes and commit
git add .
git commit -m "feat: your feature description"

# Push and create PR
git push origin feature/your-feature
```

## Useful Links

- [Full Documentation](README.md)
- [API Documentation](docs/API.md)
- [Contributing Guide](docs/CONTRIBUTING.md)
- [GitHub Repository](https://github.com/Satrajeeth/AgroVision-AI)

## Support

- 📧 Open an issue on GitHub
- 💬 Check existing documentation
- 🤝 Join discussions

---

Last Updated: November 2024
