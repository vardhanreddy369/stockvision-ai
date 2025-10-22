# Contributing to StockVision AI

We welcome contributions from the open-source community. This document outlines the process and expectations for contributing to the StockVision AI project.

---

## Code of Conduct

All contributors are expected to maintain professional conduct and treat fellow contributors with respect. Disagreements should be resolved constructively through discussion.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment management tools (venv or conda)

### Development Environment Setup

**Step 1: Fork and Clone**
```bash
git clone https://github.com/YOUR_USERNAME/stockvision-ai.git
cd stockvision-ai
```

**Step 2: Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

**Step 3: Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate
```

**Step 4: Install Dependencies**
```bash
pip install -r requirements.txt
pip install pytest black flake8
```

**Step 5: Verify Setup**
```bash
pytest tests/ -v
```

---

## Contribution Workflow

### 1. Code Quality Standards

- Follow PEP 8 style guidelines
- Add comprehensive docstrings to all functions and classes
- Keep functions focused and under 50 lines when possible
- Use type hints for function parameters and returns

### 2. Testing Requirements

All code contributions must include tests:

```bash
pytest tests/ -v --cov=src
```

Tests must pass with 100% coverage for new code.

### 3. Commit Message Format

Use conventional commit messages:

| Type | Purpose |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation updates |
| `style` | Code formatting (no logic changes) |
| `refactor` | Code restructuring |
| `test` | Test additions/modifications |
| `chore` | Build/dependency updates |

**Example:**
```
feat: implement correlation matrix visualization

- Add correlation calculation module
- Create interactive heatmap display
- Update analytics dashboard with new visualization
- Add 5 new unit tests for correlation logic
```

### 4. Pull Request Process

1. **Ensure tests pass:**
   ```bash
   pytest tests/ -v
   black src/ app/ tests/
   flake8 src/ app/ tests/
   ```

2. **Create pull request with:**
   - Clear title describing the change
   - Detailed description of modifications
   - References to related issues (if applicable)
   - Screenshots for UI changes

3. **Address review feedback**
   - Respond to all comments
   - Make requested changes in new commits
   - Re-request review after updates

---

## Code Style Guide

### Python Style

```python
# Good: Clear variable names and docstrings
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """
    Calculate Sharpe ratio for a returns series.
    
    Parameters:
        returns: Series of asset returns
        risk_free_rate: Risk-free rate of return
        
    Returns:
        Sharpe ratio value
    """
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()
```

### Documentation

- Add docstrings to all public functions
- Include parameter types and descriptions
- Provide return value documentation
- Add example usage for complex functions

---

## Testing Guidelines

### Test Structure

```python
import unittest
import pandas as pd
from src.module import function

class TestModuleName(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = pd.DataFrame(...)
    
    def test_function_with_valid_input(self):
        """Test function behavior with valid input"""
        result = function(self.test_data)
        self.assertEqual(result, expected_value)
    
    def test_function_with_edge_case(self):
        """Test function behavior with edge case"""
        result = function(edge_case_data)
        self.assertIsNotNone(result)
```

### Coverage Requirements

- Minimum 85% code coverage
- All critical paths tested
- Edge cases and error conditions included

---

## Issue Reporting

### Bug Reports

Include the following information:

- Clear description of the issue
- Steps to reproduce the problem
- Expected vs. actual behavior
- Environment details (OS, Python version)
- Relevant error messages or logs

### Feature Requests

Provide:
- Clear use case explanation
- Desired functionality description
- Potential implementation approach
- Example usage scenarios

---

## Documentation Standards

- Keep README.md current with all changes
- Update CONTRIBUTING.md for process changes
- Add inline comments for complex logic
- Document new configuration options
- Include example usage for new features

---

## Development Tools

### Code Formatting

```bash
# Format code with Black
black src/ app/ tests/

# Check style with Flake8
flake8 src/ app/ tests/ --max-line-length=100
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Release Process

Maintainers follow semantic versioning:

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Releases are tagged on main branch with version number.

---

## Communication Channels

- **Issues:** Use GitHub Issues for bugs and features
- **Discussions:** Use GitHub Discussions for questions
- **Email:** Contact maintainers for sensitive matters

---

## Recognition

Contributors will be recognized in:
- Commit history
- Release notes
- Contributors section in documentation

---

## Questions and Support

For questions about contributing, open an issue with the label `question` or contact the maintainers directly.

Thank you for contributing to StockVision AI.
