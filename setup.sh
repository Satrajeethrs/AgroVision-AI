#!/bin/bash

# AgroVision-AI Setup Script
# This script automates the setup process for the project

set -e  # Exit on error

echo "=========================================="
echo "  AgroVision-AI Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created (please update with your API keys if needed)"
else
    echo "✓ .env file already exists"
fi
echo ""

# Create logs directory
if [ ! -d "logs" ]; then
    echo "Creating logs directory..."
    mkdir -p logs
    echo "✓ logs directory created"
else
    echo "✓ logs directory already exists"
fi
echo ""

# Check if models exist
echo "Checking for trained models..."
if [ ! -f "models/model.pkl" ] || [ ! -f "models/scaler.pkl" ]; then
    echo "⚠ Crop recommendation models not found"
    echo "  Run: python src/core/train_model.py"
else
    echo "✓ Crop recommendation models found"
fi

if [ ! -f "models/plant_disease_model_final.h5" ]; then
    echo "⚠ Plant disease model not found"
    echo "  Run: python src/core/train_plant_disease_model.py"
else
    echo "✓ Plant disease model found"
fi
echo ""

# Run tests
echo "Would you like to run tests? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Running tests..."
    pytest tests/ -v
    echo ""
fi

echo "=========================================="
echo "  Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Train models (if needed):"
echo "     - python src/core/train_model.py"
echo "     - python src/core/train_plant_disease_model.py"
echo "  3. Start the application: python app.py"
echo "  4. Open http://localhost:5001 in your browser"
echo ""
echo "For more information, see README.md"
echo ""
