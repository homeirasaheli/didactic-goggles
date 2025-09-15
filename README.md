# Mini Stock Analyst  
**An AI-powered Web App for Stock Market Analysis, Technical Indicators, Forecasting, and Voice Interaction (Yahoo Finance + Tehran Stock Exchange)**  

---

## 🌍 English Description  

### 📌 Overview
**Mini Stock Analyst** is a web-based platform that combines **financial data analysis**, **technical indicators**, **machine learning predictions**, and **AI-powered insights**.  
It supports both **Yahoo Finance** and the **Tehran Stock Exchange (TSE)**, making it suitable for both international and Persian stock markets.  

This project also integrates **voice capabilities** using **Whisper (STT)** and **pyttsx3 (TTS)**, allowing users to **speak queries and receive AI-powered voice responses in Persian**.  

---

### 🚀 Features
- 📊 **Stock Data Sources**
  - Yahoo Finance (global markets)
  - Tehran Stock Exchange (TSE, Persian symbols)
- 🔎 **Technical Indicators**
  - SMA, EMA, RSI, MACD, Bollinger Bands
- 🤖 **AI-Powered Analysis**
  - GPT-based insights ("Lama" section)
- 📈 **Machine Learning Forecasting**
  - RandomForest-based prediction (DeepSeek module)
- 🎤 **Speech Integration**
  - **STT (Speech-to-Text)** using Whisper
  - **TTS (Text-to-Speech)** with pyttsx3 (Persian voice supported)
- 🖥 **Interactive Dashboard**
  - Line & candlestick charts
  - Volume visualization
  - Dark-mode optimized UI with Chart.js

---

### 🛠 Tech Stack
- **Backend**: FastAPI, Python  
- **Data**: Yahoo Finance API, pytse-client  
- **Machine Learning**: Scikit-learn (RandomForest), Indicators with Pandas/Numpy  
- **AI Integration**: OpenAI GPT (customizable model, default: `gpt-4o-mini`)  
- **Speech Processing**: Whisper (STT), pyttsx3 (TTS), ffmpeg  
- **Frontend**: HTML, CSS, JavaScript, Chart.js  

---

### ⚙️ Installation & Run
```bash
# Clone the repository
git clone https://github.com/your-username/mini-stock-analyst.git
cd mini-stock-analyst

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
App will be available at: http://127.0.0.1:8082/
