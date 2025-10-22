# 🚀 Deployment & Live Links

## Live Applications

### 📊 Interactive Dashboard
**URL:** https://stockvision-ai-hglrftaeu7t3wxxis5hyjs.streamlit.app

This is the **main application** with:
- Real-time stock analysis
- LSTM price predictions
- GradientBoosting direction signals
- Portfolio performance metrics
- Backtesting engine
- Market analytics and correlations

**Access:** Direct link to live dashboard

---

### 🌐 Landing Page & Documentation
**URL:** https://vardhanreddy369.github.io/stockvision-ai/

This is the **GitHub Pages site** with:
- Project overview
- Feature descriptions
- Technology stack information
- Links to the interactive dashboard
- Community guidelines

**Access:** GitHub Pages static site

---

## Repository Links

- **Main Repository:** https://github.com/vardhanreddy369/stockvision-ai
- **Main Branch:** https://github.com/vardhanreddy369/stockvision-ai/tree/main
- **Feature Branch:** https://github.com/vardhanreddy369/stockvision-ai/tree/feature/ui-improvements

---

## Deployment Platforms

### Streamlit Cloud
- **Service:** Streamlit Cloud (streamlit.io/cloud)
- **Repository:** Connected to GitHub repo
- **Branch:** `main`
- **Main File:** `app/app.py`
- **Status:** ✅ Live and running

### GitHub Pages
- **Service:** GitHub Pages
- **Source:** `/docs` folder
- **Configuration:** Main branch, /docs folder
- **Status:** ✅ Live and running

---

## How to Deploy Locally

### Prerequisites
```bash
Python 3.8+
pip
```

### Installation
```bash
# Clone repository
git clone https://github.com/vardhanreddy369/stockvision-ai.git
cd stockvision-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run app/app.py
```

App will be available at: `http://localhost:8501`

---

## Environment Details

### Streamlit Cloud
- **Python Version:** 3.13.9
- **Platform:** Linux
- **Memory:** Standard tier
- **Deployment Time:** ~2-3 minutes

### Local Machine
- **Python Version:** 3.8+
- **OS:** macOS, Linux, Windows
- **Memory:** Minimum 2GB RAM recommended

---

## Troubleshooting

### App Won't Load
1. Check Streamlit Cloud logs
2. Verify all dependencies installed
3. Try restarting app via Settings → Reboot

### Dependencies Error
- Update `requirements.txt`
- Push changes to GitHub
- Reboot app on Streamlit Cloud

### Performance Issues
- Check system resources
- Reduce number of tickers analyzed
- Increase Streamlit Cloud compute tier

---

## Monitoring & Maintenance

### Health Checks
- Dashboard loads successfully
- All tabs functional
- No error messages
- Real-time data updating

### Updates
- Push code changes to GitHub
- Streamlit Cloud auto-deploys
- No manual deployment needed

---

## Support & Documentation

- 📖 **README:** Project overview and features
- 🤝 **CONTRIBUTING.md:** How to contribute
- 📝 **LICENSE:** MIT License terms
- 💬 **GitHub Issues:** Report bugs and suggest features

---

**Last Updated:** October 21, 2025
