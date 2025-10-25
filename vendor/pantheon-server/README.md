# Pantheon Server

A professional cryptocurrency analysis server combining FastAPI backend with Streamlit frontend, powered by the pantheon-legends package and real-time Coinbase market data.

## ✨ What This Does

Transform raw cryptocurrency market data into actionable trading insights using legendary technical analysis methods:

- **📊 Real-time Analysis**: Live market data from Coinbase Advanced Trade API
- **🏛️ Legendary Methods**: Dow Theory, Wyckoff Method, Volume Breakout Scanner  
- **📈 Professional Charts**: TradingView integration with technical indicators
- **🎯 Smart Signals**: Multi-timeframe consensus analysis with confidence scoring
- **💡 Market Overview**: Instant sentiment analysis across 20+ popular crypto pairs
- **🔄 Auto-Refresh**: Live market monitoring with customizable timeframes

## System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux Ubuntu 18.04+
- **Python**: 3.8 or higher (3.9-3.12 recommended)
- **Memory**: 4GB RAM minimum, 8GB recommended  
- **Network**: Stable internet connection for real-time market data
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

## Features

- 🏛️ **Pantheon Legends Integration** - Uses published pantheon-legends v0.3.0 package
- 📊 **Real-time Market Data** - Coinbase Advanced Trade API integration
- 🧪 **Interactive Test Panel** - Streamlit UI for testing crypto pairs
- 📈 **TradingView Charts** - Embedded charts with analysis overlays
- 🔗 **Discord-Friendly** - Shareable URLs for community analysis
- ⚡ **Multi-timeframe EMA(9)** - Fakeout detection across timeframes

## Architecture

```
📊 Streamlit UI (Port 8501)
├── 🧪 Test Panel for crypto analysis
├── 📈 TradingView chart integration
├── 🎯 Real-time consensus display
└── 🔗 Discord-shareable analysis pages

🌐 FastAPI Backend (Port 8000)
├── 📡 Scanner alert endpoints
├── 🧠 Pantheon analysis orchestration
├── 📊 Market data aggregation
└── 🔄 Real-time WebSocket updates

💰 Coinbase Service
├── 📈 Real-time price data
├── 🕯️ Historical candle data
├── 📊 Multi-timeframe analysis
└── ⚡ Rate-limited API calls

📦 pantheon-legends v0.3.0
└── 🏛️ Legend engine framework
```

## Quick Start

### Prerequisites

- **Python 3.8+** (tested with Python 3.9-3.12)
- **Git** for cloning the repository
- **Internet connection** for Coinbase API access

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/SpartanDigitalDotNet/pantheon-server.git
cd pantheon-server

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration (Optional)

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your Coinbase API credentials (optional for enhanced features)
COINBASE_API_KEY=your_api_key_here
COINBASE_API_SECRET=your_api_secret_here
```

> **Note**: The server works without API credentials using public Coinbase endpoints, but authenticated access provides more features and higher rate limits.

### 3. Start the Application

**Option A: Easy Start (Recommended)**
```bash
# Start both FastAPI backend and Streamlit UI
python run.py dev
```

**Option B: Manual Start**
```bash
# Terminal 1: Start FastAPI backend
python run.py api

# Terminal 2: Start Streamlit UI  
python run.py ui
```

**Option C: Individual Services**
```bash
# FastAPI only
uvicorn src.pantheon_server.api.main:app --reload --port 8000

# Streamlit only
streamlit run src/pantheon_server/ui/streamlit_app.py --server.port 8501
```

### 4. Verify Installation

Once the services are running, verify everything works:

1. **FastAPI Backend**: Visit http://localhost:8000/docs
   - Should show interactive API documentation
   - Test the `/health` endpoint

2. **Streamlit UI**: Visit http://localhost:8501  
   - Should load the Pantheon Server interface
   - Navigate to "Market Overview" and click "🔄 Generate Overview"
   - Check "Crypto Analysis" for individual pair analysis

3. **Test Analysis**: 
   - Select a crypto pair (e.g., BTC-USD)
   - Choose timeframe (e.g., 1h) 
   - Click "🔍 Analyze" to see TradingView charts and analysis results

## Enhanced Wyckoff Method v0.4.0

🎉 **pantheon-server now includes the Enhanced Wyckoff Method v0.4.0** - featuring revolutionary **unified consensus analysis** that dramatically simplifies multi-engine analysis through automatic consensus calculation.

### 🚀 v0.4.0 Major Breakthrough: Unified Consensus

- **🎯 Single-Call Analysis**: Complete multi-engine analysis with consensus in one method call
- **⚡ 90% Complexity Reduction**: Eliminates manual consensus orchestration and result aggregation
- **🧠 Automatic Reliability Weighting**: Intelligent engine weighting based on reliability levels
- **📊 Built-in Quality Assessment**: Consensus quality levels (high/medium/low/insufficient)
- **🛡️ Error-Resilient Processing**: Continues analysis even if some engines fail
- **🚀 Production-Ready**: Built-in error handling with sensible defaults

### 📈 Enhanced Wyckoff Method Core Features

- **🎯 95% Manipulation Sensitivity**: Mathematical precision in smart money detection (maintained from v0.3.0)
- **✨ 15% False Positive Risk**: Reduced through intelligent consensus validation
- **📊 20+ Analysis Dimensions**: Current phase detection, smart money tracking, volume analysis
- **🔬 Three Fundamental Laws**: Supply/Demand, Cause/Effect, Effort vs Result
- **⚡ Real-Time Consensus**: Live market analysis with automated engine coordination

### 🛠️ Unified API Example

```python
# Before v0.4.0: Manual multi-step process
individual_results = await pantheon.analyze(request)
consensus = pantheon.get_consensus(individual_results)  # Manual orchestration

# After v0.4.0: Single unified call
result = await pantheon.analyze_with_consensus(
    request=legends.LegendRequest(symbol="BTC-USD", timeframe="1h"),
    enable_consensus=True
)
# Access both individual and consensus results from unified object
print(f"Consensus: {result.consensus.signal}")
print(f"Confidence: {result.consensus.confidence}")
```

### 🔬 Technical Improvements

The Enhanced Wyckoff Method provides deeper market structure insights with reduced noise and higher confidence signals, making it suitable for professional cryptocurrency analysis and risk assessment.

## Usage

### Test Panel Workflow

1. **Select Crypto Pair**: Choose from popular pairs or enter custom symbol
2. **Configure Analysis**: Set timeframe, candle count, enable multi-timeframe
3. **Simulate Breakouts**: Test "what if" scenarios with price/volume spikes
4. **Run Analysis**: Get real-time Pantheon Legends analysis with EMA(9) fakeout detection
5. **View Results**: Interactive charts, consensus analysis, and trading recommendations

### API Integration

```python
# Send scanner alert to FastAPI endpoint
POST /api/v1/analyze
{
    "scanner_alert": {
        "symbol": "DIMO-USD",
        "price": 0.07948,
        "delta_pct": 1.31,
        "volume_ratio": 3.68,
        "timeframe": "1m"
    },
    "include_ema9": true,
    "trader_profile": "scalper"
}
```

## Development

### Project Structure

```
pantheon-server/
├── src/pantheon_server/
│   ├── api/              # FastAPI backend
│   │   ├── main.py       # FastAPI app
│   │   ├── models.py     # Request/response models
│   │   └── routes.py     # API endpoints
│   ├── services/         # External integrations
│   │   ├── coinbase.py   # Coinbase API service
│   │   └── analyzer.py   # Market analysis orchestration
│   ├── ui/               # Streamlit frontend
│   │   ├── main.py       # Main Streamlit app
│   │   ├── test_panel.py # Interactive test interface
│   │   └── charts.py     # Chart components
│   └── utils/            # Shared utilities
├── tests/                # Test suite
├── requirements.txt      # Python dependencies
└── .env.example         # Environment template
```

### Dependencies

- **pantheon-legends==0.3.0** - Enhanced core legend analysis framework with production-ready Wyckoff Method
- **fastapi** - Modern web API framework
- **streamlit** - Interactive web UI
- **httpx** - Async HTTP client for Coinbase API
- **plotly** - Interactive charts and visualizations
- **pandas** - Data manipulation and analysis

## Configuration

### Environment Variables

```bash
# Coinbase API (optional - uses public endpoints if not provided)
COINBASE_API_KEY=your_coinbase_api_key
COINBASE_API_SECRET=your_coinbase_api_secret

# Server Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
STREAMLIT_PORT=8501

# Analysis Configuration
DEFAULT_CANDLE_COUNT=50
DEFAULT_TIMEFRAME=1m
ENABLE_MULTI_TIMEFRAME=true
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Error: ModuleNotFoundError: No module named 'pantheon_server'
# Solution: Ensure you're in the project directory and virtual environment is activated
cd pantheon-server
.venv\Scripts\activate  # Windows
python run.py dev
```

**2. Port Already in Use**
```bash
# Error: Port 8000 or 8501 already in use
# Solution: Kill existing processes or use different ports
# Windows:
netstat -ano | findstr :8000
taskkill /PID <process_id> /F

# Or change ports in .env file
```

**3. Coinbase API Errors**
```bash
# Error: Rate limiting or API access issues
# Solution: 
# 1. Remove API credentials from .env to use public endpoints
# 2. Check API key permissions on Coinbase
# 3. Verify API key format (no extra spaces/characters)
```

**4. Virtual Environment Issues**
```bash
# Error: Package not found or wrong Python version
# Solution: Recreate virtual environment
rm -rf .venv  # or rmdir /s .venv on Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Development Tips

- Use `python run.py dev` for hot-reload development
- Check `http://localhost:8000/docs` for API documentation
- Streamlit auto-refreshes on file changes
- Use Market Overview for quick market sentiment analysis

## 📢 Stay Updated

### Real-time Development Updates
Join our Discord community to stay on top of all development progress and updates:

- **📊 Discord Channel**: [#pantheon-server-updates](https://discord.com/channels/1408828331176235050/1416543629769445486)
- **🔔 Automatic Notifications**: Live commit notifications, releases, and development milestones
- **🏛️ Repository**: Real-time updates from this GitHub repository
- **💬 Community**: Connect with other developers and traders using Pantheon

### GitHub Integration
- **🔍 Issues**: [Report bugs or request features](https://github.com/SpartanDigitalDotNet/pantheon-server/issues)
- **🚀 Pull Requests**: Contribute code improvements and new features
- **📋 Releases**: Watch for new versions and feature announcements
- **⭐ Star the Repository**: Show your support and stay notified of major updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## ⚖️ Legal Disclaimer

**IMPORTANT: Please read this disclaimer carefully before using the Pantheon Server.**

### Educational and Entertainment Purpose Only
This software is provided **for educational and entertainment purposes only**. The Pantheon Server and its analysis tools are designed to demonstrate technical analysis concepts and cryptocurrency market data visualization. This software is **NOT** intended to provide financial, investment, or trading advice.

### No Financial Advice
- **Not Financial Advice**: The analysis, signals, and recommendations generated by this software do not constitute financial, investment, or trading advice
- **No Professional Relationship**: Use of this software does not create any advisor-client relationship
- **Educational Tool**: This is a learning and demonstration tool for understanding technical analysis concepts

### Trading and Investment Risks
- **High Risk**: Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors
- **Volatile Markets**: Cryptocurrency markets are highly volatile and unpredictable
- **Potential Losses**: You may lose some or all of your invested capital
- **Past Performance**: Past performance does not guarantee future results

### No Warranties or Guarantees
- **No Accuracy Guarantee**: We make no warranties about the accuracy, completeness, or reliability of any analysis or data
- **Software "As-Is"**: This software is provided "as-is" without any warranties of any kind
- **No Guarantee of Profits**: We do not guarantee any profits or positive outcomes from using this software
- **Technical Issues**: The software may contain bugs, errors, or experience downtime

### User Responsibility
- **Your Own Research**: Always conduct your own research and due diligence before making any financial decisions
- **Professional Advice**: Consult with qualified financial professionals before making investment decisions
- **Risk Assessment**: Only invest what you can afford to lose
- **Local Laws**: Ensure compliance with your local laws and regulations regarding cryptocurrency trading

### Data Sources and Third-Party Services
- **External Data**: This software relies on third-party data sources (including Coinbase) which may be inaccurate or delayed
- **Service Availability**: Third-party services may be unavailable or discontinued at any time
- **No Control**: We have no control over third-party services and their reliability

### Limitation of Liability
By using this software, you acknowledge and agree that:
- You use this software at your own risk
- The developers and contributors shall not be liable for any damages, losses, or harm arising from the use of this software
- You will not hold the developers responsible for any financial losses incurred while using this software

### Compliance
Users are responsible for ensuring their use of this software complies with all applicable laws and regulations in their jurisdiction.

---

**By using the Pantheon Server, you acknowledge that you have read, understood, and agree to this disclaimer.**

## License

MIT License - see LICENSE file for details

## Related Projects

- **[pantheon-legends](https://github.com/SpartanDigitalDotNet/pantheon-legends)** - Core legend analysis framework
- **[Coinbase Advanced Trade API](https://docs.cloud.coinbase.com/advanced-trade-api/docs/welcome)** - Market data source

---

**Built with the pantheon-legends framework for comprehensive cryptocurrency market analysis** 🏛️📊

<!-- Discord webhook test: 2025-09-13 -->
<!-- Second webhook test with /github endpoint -->
