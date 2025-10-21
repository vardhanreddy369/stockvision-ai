# Contributing to StockVision AI

Thank you for your interest in contributing to StockVision AI! We welcome contributions from the community.

## 📋 Code of Conduct

Please be respectful and constructive in all interactions with other contributors and maintainers.

## 🚀 How to Contribute

### 1. Fork the Repository
```bash
git clone https://github.com/YOUR_USERNAME/stockvision-ai.git
cd stockvision-ai
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes
- Write clean, readable code
- Add docstrings to functions
- Follow PEP 8 style guide
- Add tests for new features

### 4. Commit Your Changes
```bash
git commit -m "feat: add your feature description"
```

### 5. Push to Your Fork
```bash
git push origin feature/your-feature-name
```

### 6. Create a Pull Request
- Go to the original repository
- Click "New Pull Request"
- Provide a clear description of your changes
- Reference any related issues

## 🧪 Testing

Before submitting a PR, please ensure all tests pass:

```bash
pytest tests/ -v
```

## 📝 Commit Message Guidelines

Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `style:` for code style changes
- `refactor:` for code refactoring
- `test:` for test additions/modifications

Example:
```
feat: add correlation matrix visualization

- Implement correlation calculation
- Add heatmap display
- Update analytics section
```

## 🐛 Reporting Issues

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)

## 💡 Feature Requests

For feature requests:
- Explain the use case
- Describe the desired behavior
- Provide examples if applicable

## 📚 Documentation

- Update README.md if you change functionality
- Add docstrings to all functions
- Comment complex logic
- Keep documentation up to date

## ✅ Pull Request Checklist

Before submitting a PR:
- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] Code follows PEP 8 style guide
- [ ] Docstrings added for new functions
- [ ] README updated if needed
- [ ] Commit messages follow guidelines
- [ ] No unnecessary dependencies added

## 🎯 Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest black flake8

# Run tests
pytest tests/ -v

# Format code
black src/ app/ tests/

# Check style
flake8 src/ app/ tests/
```

## 📞 Questions?

Feel free to open an issue with the label `question` or reach out to the maintainers.

---

Thank you for contributing! 🎉
