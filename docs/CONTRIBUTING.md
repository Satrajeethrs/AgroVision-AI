# Contributing to AgroVision-AI

First off, thank you for considering contributing to AgroVision-AI! It's people like you that make this project such a great tool for farmers and agriculture enthusiasts.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, virtualenv, or conda)
- Basic understanding of Flask, TensorFlow, and scikit-learn

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AgroVision-AI.git
   cd AgroVision-AI
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/Satrajeeth/AgroVision-AI.git
   ```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected behavior**
- **Actual behavior**
- **Screenshots** (if applicable)
- **Environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description**
- **Use case** and why it would be useful
- **Possible implementation** approach (if you have ideas)
- **Examples** from other projects (if applicable)

### Your First Code Contribution

Unsure where to begin? Look for issues labeled:
- `good first issue` - Simple issues perfect for newcomers
- `help wanted` - Issues where we need help
- `documentation` - Documentation improvements

### Pull Requests

1. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our style guidelines

3. **Test your changes**:
   ```bash
   pytest tests/
   ```

4. **Commit your changes** with clear messages

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub

## Development Setup

### 1. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

### 3. Set Up Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

### 4. Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_validator.py
```

## Style Guidelines

### Python Code Style

We follow **PEP 8** style guidelines. Key points:

- **Indentation**: 4 spaces (no tabs)
- **Line length**: Maximum 100 characters
- **Imports**: Group stdlib, third-party, and local imports
- **Naming conventions**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`

### Example

```python
"""Module docstring describing the module."""

import os
from typing import List, Dict

import pandas as pd
from flask import Flask

from src.utils.validation import validate_input


class CropRecommender:
    """Class docstring describing the class.
    
    Attributes:
        model: The trained machine learning model
        scaler: The feature scaler
    """
    
    def __init__(self, model_path: str):
        """Initialize the recommender.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model = self._load_model(model_path)
    
    def predict(self, features: Dict[str, float]) -> str:
        """Make a crop prediction.
        
        Args:
            features: Dictionary of input features
            
        Returns:
            The recommended crop name
            
        Raises:
            ValueError: If features are invalid
        """
        # Implementation
        pass
```

### Documentation

- **Docstrings**: Use Google-style docstrings
- **Comments**: Explain why, not what
- **README**: Update if you add new features
- **API docs**: Document all public APIs

### Type Hints

Use type hints for function parameters and return values:

```python
def calculate_score(n: float, p: float, k: float) -> float:
    """Calculate soil quality score."""
    return (n + p + k) / 3
```

## Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(crop): Add support for wheat crop recommendation

Implement wheat-specific logic for crop recommendation
including optimal NPK ranges and climate requirements.

Closes #123
```

```
fix(disease): Correct disease classification threshold

Change the confidence threshold from 0.7 to 0.8 to reduce
false positives in disease detection.

Fixes #456
```

## Pull Request Process

### Before Submitting

1. ✅ Ensure all tests pass
2. ✅ Update documentation if needed
3. ✅ Add tests for new features
4. ✅ Follow the style guidelines
5. ✅ Rebase on latest main branch

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe the tests you ran

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
```

### Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged
4. Celebrate! 🎉 You're now a contributor!

## Testing Guidelines

### Writing Tests

```python
import pytest
from src.utils.validation import validate_input

def test_validate_input_valid():
    """Test validation with valid inputs."""
    result = validate_input({'N': 90, 'P': 42, 'K': 43})
    assert result is True

def test_validate_input_invalid():
    """Test validation with invalid inputs."""
    with pytest.raises(ValueError):
        validate_input({'N': -10, 'P': 42, 'K': 43})
```

### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Include integration tests for critical paths

## Questions?

Feel free to:
- Open an issue for questions
- Join our discussions on GitHub
- Contact the maintainers directly

## Recognition

Contributors will be recognized in:
- README acknowledgments
- Release notes
- GitHub contributors page

Thank you for contributing to AgroVision-AI! 🌾
