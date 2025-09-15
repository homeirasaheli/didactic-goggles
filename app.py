
from __future__ import annotations
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()
# ================================
# Imports
# ================================
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pytse_client as tse
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------- Optional: yfinance ----------
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

# ---------- OpenAI ----------
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def get_openai_client() -> Optional[OpenAI]:
    """Create OpenAI client lazily only if API key exists."""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return None
    return OpenAI(api_key=key)

# TTS with pyttsx3
try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False
    logging.warning("pyttsx3 not installed. Install with: pip install pyttsx3")

# STT with Whisper
try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    logging.warning("Whisper not installed. Install with: pip install git+https://github.com/openai/whisper.git")

# Check for ffmpeg
try:
    import subprocess
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    HAS_FFMPEG = True
except (subprocess.CalledProcessError, FileNotFoundError):
    HAS_FFMPEG = False
    logging.warning("ffmpeg not installed or not found in PATH. Install ffmpeg for Whisper support.")

# ================================
# App & Logging
# ================================
app = FastAPI(title="Mini Stock Analyst", version="0.6.3")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mini-stock-analyst")

# Create temp directory
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# ================================
# Health check
# ================================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "Mini Stock Analyst",
        "version": app.version,
        "yfinance": HAS_YF,
        "tts": HAS_TTS,
        "whisper": HAS_WHISPER,
        "ffmpeg": HAS_FFMPEG,
        "openai_model": OPENAI_MODEL,
        "openai_key": bool(OPENAI_API_KEY),
    }

# (Optional) Prevent 404 for favicon
@app.get("/favicon.ico")
def favicon():
    from fastapi import Response
    return Response(status_code=204)

# ================================
# Generate synthetic stock data for demo
# ================================
@app.get("/generate_series")
async def generate_series(symbol: str = "TEST", n: int = 120, seed: int = 42, start: float = 100.0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.05/252, scale=1.0, size=n)
    prices = [start]
    for s in steps:
        prices.append(prices[-1] * (1 + s/100))
    close = np.array(prices[1:]).round(4)
    base = pd.Series(close)
    o = base.shift(1).fillna(base.iloc[0])
    h = pd.concat([o, base], axis=1).max(axis=1) + np.random.rand(n)*0.5
    l = pd.concat([o, base], axis=1).min(axis=1) - np.random.rand(n)*0.5
    v = (np.abs(np.random.randn(n))*1e6).round().astype(int)
    dates = list(range(1, n+1))
    ohlc = {
        "open": o.round(4).tolist(),
        "high": h.round(4).tolist(),
        "low": l.round(4).tolist(),
        "close": base.round(4).tolist(),
    }
    return {
        "symbol": symbol,
        "dates": dates,
        "close": base.round(4).tolist(),
        "ohlc": ohlc,
        "volume": v.tolist(),
    }

# ================================
# Get real stock data from Yahoo Finance
# ================================
logger_prices = logging.getLogger("prices")

@app.get("/prices")
async def get_prices(symbol: str, period: str = "1y", interval: str = "1d"):
    if not HAS_YF:
        raise HTTPException(500, detail="yfinance not installed. Run: pip install yfinance")
    try:
        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as e:
        logger_prices.exception("Yahoo fetch error")
        raise HTTPException(502, detail=f"Yahoo fetch error: {e}")
    if df is None or df.empty:
        raise HTTPException(404, detail=f"No data returned for {symbol}")

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join([str(c) for c in col if c]) for col in out.columns.values]

    def find_col(name: str):
        lname = name.lower()
        for c in out.columns:
            if str(c).strip().lower() == lname:
                return c
        for c in out.columns:
            if lname in str(c).strip().lower():
                return c
        return None

    open_col = find_col("open") or "Open"
    high_col = find_col("high") or "High"
    low_col = find_col("low") or "Low"
    close_col = find_col("close") or "Close"
    vol_col = find_col("volume") or "Volume"

    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.dropna(subset=[close_col])
    out = out.sort_index()

    dates = out.index.strftime("%Y-%m-%dT%H:%M:%S").tolist()

    def series_or_none(col):
        if col in out:
            s = pd.to_numeric(out[col], errors="coerce").astype(float).round(4)
            return [x if pd.notna(x) else None for x in s.tolist()]
        return None

    ohlc = {
        "open": series_or_none(open_col),
        "high": series_or_none(high_col),
        "low": series_or_none(low_col),
        "close": series_or_none(close_col),
    }
    volume = None
    if vol_col in out:
        vv = pd.to_numeric(out[vol_col], errors="coerce").astype(float)
        volume = [int(round(x)) if pd.notna(x) else None for x in vv.tolist()]

    return {
        "symbol": symbol,
        "dates": dates,
        "close": ohlc["close"],
        "ohlc": ohlc,
        "volume": volume
    }

# ================================
# Tehran Stock Exchange (TSETMC)
# ================================
logger_tse = logging.getLogger("mini-stock-analyst")

def norm_txt(s: str) -> str:
    """Normalize Persian characters."""
    return s.replace("ي", "ی").replace("ك", "ک").strip()

@app.get("/tse/prices")
async def tse_prices(symbol: str, adjust: bool = True, include_jdate: bool = True):
    symbol = norm_txt(symbol)
    try:
        # Removed timeout param since pytse_client.download() does not accept it
        data = tse.download(symbols=symbol, write_to_csv=False)
    except Exception as e:
        logger_tse.exception("TSE fetch error (download)")
        raise HTTPException(502, detail=f"TSE fetch error: {e}. Ensure no proxy redirect to localhost and check network connectivity.")

    if not isinstance(data, dict) or not data:
        raise HTTPException(404, detail=f"Empty response from TSE for {symbol}")

    df = data.get(symbol)
    if df is None:
        for k in data.keys():
            if norm_txt(k) == symbol:
                df = data[k]
                break

    if df is None or df.empty:
        raise HTTPException(404, detail=f"No data returned for {symbol}. Available symbols: {list(data.keys())}")

    df = df.copy()

    def pick(col_name: str):
        target = col_name.lower()
        for c in df.columns:
            if str(c).strip().lower() == target:
                return c
        for c in df.columns:
            if target in str(c).strip().lower():
                return c
        return None

    open_col = pick("open") or pick("priceopen") or pick("first")
    high_col = pick("high") or pick("max") or pick("highprice")
    low_col = pick("low") or pick("min") or pick("lowprice")
    close_col = pick("close") or pick("last") or pick("closeprice")
    vol_col = pick("volume") or pick("vol") or pick("value") or pick("tradevolume")

    if include_jdate and (pick("jdate") or "jdate" in df.columns):
        jcol = pick("jdate") or "jdate"
        date_series = df[jcol].astype(str)
    else:
        dcol = pick("date") or "date"
        if dcol not in df.columns:
            raise HTTPException(404, detail=f"No date column. Columns: {list(df.columns)}")
        dts = pd.to_datetime(df[dcol], errors="coerce")
        df = df.assign(_date=dts).dropna(subset=["_date"]).sort_values("_date")
        date_series = df["_date"].dt.strftime("%Y-%m-%d")

    if close_col not in df:
        raise HTTPException(404, detail=f"No close column. Columns: {list(df.columns)}")

    close = pd.to_numeric(df[close_col], errors="coerce")
    valid_mask = close.notna()

    def safe_series(col: Optional[str]):
        if col is None or col not in df:
            return None
        s = pd.to_numeric(df[col], errors="coerce").where(valid_mask)
        return s

    open_s = safe_series(open_col)
    high_s = safe_series(high_col)
    low_s = safe_series(low_col)
    close_s = close.where(valid_mask)

    dates = date_series.where(valid_mask).dropna().astype(str).tolist()

    def list_or_none(s: Optional[pd.Series]):
        if s is None:
            return None
        return [round(float(x), 4) if pd.notna(x) else None for x in s.tolist()]

    ohlc = {
        "open": list_or_none(open_s),
        "high": list_or_none(high_s),
        "low": list_or_none(low_s),
        "close": list_or_none(close_s),
    }
    volume = None
    if vol_col in df:
        vv = pd.to_numeric(df[vol_col], errors="coerce").where(valid_mask)
        volume = [int(round(float(x))) if pd.notna(x) else None for x in vv.tolist()]

    lengths = [len(dates)] + [len(v) for v in ohlc.values() if isinstance(v, list)]
    if isinstance(volume, list):
        lengths.append(len(volume))
    L = min(lengths)

    dates = dates[-L:]
    for k in ohlc.keys():
        if isinstance(ohlc[k], list):
            ohlc[k] = ohlc[k][-L:]
    if isinstance(volume, list):
        volume = volume[-L:]

    return {
        "symbol": symbol,
        "dates": dates,
        "close": ohlc["close"],
        "ohlc": ohlc,
        "volume": volume
    }

# ================================
# (باقی کد بدون تغییر باقی می‌ماند)
# ================================

# ... شما می‌توانید بقیه کد را بدون تغییر ادامه دهید ...


# ================================
# Indicators (SMA/EMA/RSI + MACD + Bollinger)
# ================================
class IndicatorsInput(BaseModel):
    close: List[float] = Field(..., description="Close price series")
    sma_periods: List[int] = Field(default_factory=lambda: [20, 50])
    ema_periods: List[int] = Field(default_factory=lambda: [12, 26])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - sig
    return line, sig, hist

def bollinger(series: pd.Series, period: int = 20, k: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower

def handle_nan(values):
    return [x if pd.notna(x) else None for x in values]

@app.post("/indicators")
async def indicators(data: IndicatorsInput):
    close = pd.Series(data.close, dtype=float)
    out: Dict[str, object] = {"sma": {}, "ema": {}, "rsi": [], "macd": {}, "bb": {}}

    for p in sorted(set(data.sma_periods)):
        s = sma(close, p)
        out["sma"][str(p)] = handle_nan(s.round(4).tolist())

    for p in sorted(set(data.ema_periods)):
        e = ema(close, p)
        out["ema"][str(p)] = handle_nan(e.round(4).tolist())

    rr = rsi(close, data.rsi_period)
    out["rsi"] = handle_nan(rr.round(2).tolist())

    m_line, m_sig, m_hist = macd(close, data.macd_fast, data.macd_slow, data.macd_signal)
    out["macd"] = {
        "line": handle_nan(m_line.round(4).tolist()),
        "signal": handle_nan(m_sig.round(4).tolist()),
        "hist": handle_nan(m_hist.round(4).tolist()),
    }

    u, mid, l = bollinger(close, data.bb_period, data.bb_std)
    out["bb"] = {
        "upper": handle_nan(u.round(4).tolist()),
        "middle": handle_nan(mid.round(4).tolist()),
        "lower": handle_nan(l.round(4).tolist()),
    }

    return out

# ================================
# Lama-based prediction
# ================================
client = OpenAI(api_key=OPENAI_API_KEY)

class LamaInput(BaseModel):
    query: str = Field(..., description="سوال کاربر")
    indicators: Optional[Dict] = Field(None, description="داده‌های اندیکاتورها برای تحلیل")

def get_lama_response(prompt):
    try:
        chat = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3,
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("OpenAI chat error")
        return f"خطا در ارتباط با OpenAI: {e}. لطفاً بررسی کنید که کلید API معتبر است و محدودیت نرخ درخواست‌ها را چک کنید."

def get_lama_analysis(query, indicators=None):
    prompt = query
    if indicators:
        last_sma = {k: (v[-1] if isinstance(v, list) and v else None) for k, v in indicators.get('sma', {}).items()}
        last_ema = {k: (v[-1] if isinstance(v, list) and v else None) for k, v in indicators.get('ema', {}).items()}
        last_rsi = indicators.get('rsi', [])[-1] if indicators.get('rsi', []) else None
        last_macd = {k: (v[-1] if isinstance(v, list) and v else None) for k, v in indicators.get('macd', {}).items()}
        last_bb = {k: (v[-1] if isinstance(v, list) and v else None) for k, v in indicators.get('bb', {}).items()}
        prompt = (
            "داده‌های اندیکاتورها (آخرین مقادیر):\n"
            f"SMA: {last_sma}\n"
            f"EMA: {last_ema}\n"
            f"RSI: {last_rsi}\n"
            f"MACD: {last_macd}\n"
            f"Bollinger Bands: {last_bb}\n"
            f"بر اساس این داده‌ها، تحلیل جامعی به فارسی ارائه دهید: {query}"
        )

    return get_lama_response(prompt)

@app.post("/lama_predict")
async def lama_predict(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    indicators = body.get("indicators", None)
    response = get_lama_analysis(prompt, indicators)
    return {"response": response}

# ================================
# Speech Integration (STT with Whisper, TTS with pyttsx3)
# ================================
@app.post("/api/stt")
async def api_stt(file: UploadFile = File(...)):
    if not HAS_WHISPER:
        raise HTTPException(500, detail="Whisper not installed. Run: pip install git+https://github.com/openai/whisper.git")
    if not HAS_FFMPEG:
        raise HTTPException(500, detail="ffmpeg not installed. Install ffmpeg and ensure it's in the system PATH.")
    if not HAS_TTS:
        raise HTTPException(500, detail="pyttsx3 not installed. Run: pip install pyttsx3")
    try:
        tmp_path = TEMP_DIR / file.filename
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        stt_model = whisper.load_model("small")
        result = stt_model.transcribe(str(tmp_path), language="fa")
        text_in = result["text"]

        lama_reply = get_lama_response(text_in)
        out_audio = TEMP_DIR / f"reply_{os.getpid()}.mp3"
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'persian' in voice.name.lower() or 'fa' in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
        else:
            logger.warning("No Persian voice found, using default voice")
        engine.save_to_file(lama_reply, str(out_audio))
        engine.runAndWait()

        if not out_audio.exists():
            raise Exception("Failed to generate audio file")

        return JSONResponse({
            "transcription": text_in,
            "llama_text_reply": lama_reply,
            "audio_file": str(out_audio)
        })
    except Exception as e:
        logger.exception("STT error")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

@app.post("/api/tts")
async def api_tts(text: str = Form(...)):
    if not HAS_TTS:
        raise HTTPException(500, detail="pyttsx3 not installed. Run: pip install pyttsx3")
    if not text or not text.strip():
        raise HTTPException(400, detail="Text parameter is required and cannot be empty")
    try:
        out_audio = TEMP_DIR / f"tts_{os.getpid()}.mp3"
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'persian' in voice.name.lower() or 'fa' in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
        else:
            logger.warning("No Persian voice found, using default voice")
        engine.save_to_file(text.strip(), str(out_audio))
        engine.runAndWait()
        if not out_audio.exists():
            raise Exception("Failed to generate audio file")
        return JSONResponse({"audio_file": str(out_audio)})
    except Exception as e:
        logger.exception("TTS error")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/download")
async def api_download(path: str):
    full_path = TEMP_DIR / os.path.basename(path)
    if not full_path.exists():
        raise HTTPException(404, detail="File not found")
    return FileResponse(full_path, media_type="audio/mpeg", filename=full_path.name)

# ================================
# RandomForest prediction ("DeepSeek" section)
# ================================
def get_stock_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    if not HAS_YF:
        raise HTTPException(500, detail="yfinance not installed. Run: pip install yfinance")
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise HTTPException(404, detail=f"No data from Yahoo for symbol={symbol}")
    need = ["Open", "High", "Low", "Close", "Volume"]
    for c in need:
        if c not in df.columns:
            raise HTTPException(500, detail=f"Missing required column: {c}")
    return df

def rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def get_deepseek_prediction(data):
    try:
        close = pd.Series(data['close'], dtype=float)
        sma_20 = close.rolling(window=20, min_periods=20).mean()
        rsi_14 = rsi_series(close, period=14)

        features = pd.DataFrame({
            'open': pd.Series(data['ohlc']['open']).astype(float),
            'high': pd.Series(data['ohlc']['high']).astype(float),
            'low': pd.Series(data['ohlc']['low']).astype(float),
            'volume': pd.Series(data['volume']).astype(float),
            'SMA_20': sma_20,
            'RSI_14': rsi_14,
        })

        features = features.dropna()
        if features.shape[0] < 60:
            raise HTTPException(400, detail="داده کافی برای آموزش مدل وجود ندارد (حداقل ≈60 ردیف پس از محاسبه اندیکاتورها)")

        y = pd.Series(data['close']).loc[features.index].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        last_features = features.tail(1)
        pred = model.predict(last_features)[0]

        leaf_preds = np.array([t.predict(last_features)[0] for t in model.estimators_])
        pred_std = float(np.std(leaf_preds))
        return {"prediction": float(pred), "uncertainty_std": pred_std}
    except Exception as e:
        raise HTTPException(500, detail=f"خطا در پیش‌بینی DeepSeek: {e}")

@app.get("/deepseek_predict")
async def deepseek_predict(symbol: str):
    try:
        data = await tse_prices(symbol)
        result = get_deepseek_prediction(data)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("DeepSeek prediction error")
        raise HTTPException(500, detail=f"DeepSeek prediction error: {e}")

# ================================
# Frontend UI (HTML + JS)
# ================================
@app.get("/", response_class=HTMLResponse)
async def home():
    html = """
<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mini Stock Analyst</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial"></script>
  <style>
    body { 
      font-family: Vazirmatn, system-ui, sans-serif; 
      background: #1a1b26; 
      color: #c0caf5; 
      margin: 0; 
      padding: 20px; 
    }
    .container { 
      max-width: 1000px; 
      margin: 0 auto; 
    }
    .card { 
      background: #24283b; 
      border-radius: 10px; 
      padding: 20px; 
      margin-bottom: 20px; 
      box-shadow: 0 2px 10px rgba(0,0,0,0.3); 
    }
    .row { 
      display: flex; 
      gap: 15px; 
      flex-wrap: wrap; 
      align-items: flex-end; 
    }
    label { 
      font-size: 14px; 
      color: #a9b1d6; 
      margin-bottom: 5px; 
      display: block; 
    }
    input, select, button { 
      background: #2e3b55; 
      color: #c0caf5; 
      border: 1px solid #414868; 
      border-radius: 6px; 
      padding: 10px; 
      font-size: 14px; 
      transition: all 0.2s; 
    }
    input:focus, select:focus, button:focus { 
      outline: none; 
      border-color: #7aa2f7; 
      box-shadow: 0 0 5px rgba(122,162,247,0.5); 
    }
    button { 
      cursor: pointer; 
      background: #7aa2f7; 
      color: #1a1b26; 
      font-weight: bold; 
    }
    button:hover { 
      background: #89b4fa; 
    }
    .muted { 
      color: #a9b1d6; 
      font-size: 12px; 
    }
    canvas { 
      background: #16161e; 
      border-radius: 10px; 
      padding: 10px; 
    }
    .col { 
      display: flex; 
      flex-direction: column; 
      gap: 5px; 
      flex: 1; 
    }
    .mode { 
      display: flex; 
      gap: 10px; 
      align-items: center; 
    }
    pre { 
      white-space: pre-wrap; 
      background: #16161e; 
      padding: 10px; 
      border-radius: 6px; 
      color: #c0caf5; 
    }
    .recording { 
      color: #f7768e; 
      font-weight: bold; 
    }
    h2, h3 { 
      color: #7aa2f7; 
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>Mini Stock Analyst <span class="muted">تحلیل هوشمند سهام</span></h2>

    <!-- Yahoo Finance -->
    <div class="card">
      <h3>داده‌های یاهو فایننس</h3>
      <div class="row">
        <div class="col">
          <label>نماد (مثال: AAPL یا MSFT)</label>
          <input id="sym" value="AAPL" placeholder="نماد را وارد کنید" />
        </div>
        <div class="col">
          <label>بازه زمانی</label>
          <select id="period">
            <option>6mo</option>
            <option selected>1y</option>
            <option>2y</option>
            <option>5y</option>
            <option>max</option>
          </select>
        </div>
        <div class="col">
          <label>بازه نمونه‌برداری</label>
          <select id="interval">
            <option selected>1d</option>
            <option>1h</option>
            <option>30m</option>
            <option>15m</option>
            <option>5m</option>
          </select>
        </div>
        <button id="btn-real">دریافت داده</button>
      </div>
    </div>

    <!-- Tehran Stock Exchange -->
    <div class="card">
      <h3>بورس تهران</h3>
      <div class="row">
        <div class="col">
          <label>نماد (مثال: وبملت، فولاد)</label>
          <input id="sym_tse" placeholder="نماد فارسی" />
        </div>
        <button id="btn-tse">دریافت داده</button>
      </div>
      <div class="muted" style="margin-top:8px">پشتیبانی از تاریخ شمسی و تعدیل قیمت‌ها</div>
    </div>

    <!-- Indicators & Chart -->
    <div class="card">
      <div class="row" style="justify-content:space-between;">
        <h3>نمودار و اندیکاتورها</h3>
        <div class="mode">
          <span class="muted">نمایش:</span>
          <label><input type="radio" name="mode" value="line" checked> خطی</label>
          <label><input type="radio" name="mode" value="candle"> کندل</label>
        </div>
      </div>
      <button id="btn-ind">محاسبه اندیکاتورها</button>
      <canvas id="ch" height="220"></canvas>
      <div class="muted" style="margin-top:6px">اندیکاتورها: SMA20/50, EMA12/26, RSI14, MACD, Bollinger</div>
      <h4 style="margin-top:14px">حجم معاملات</h4>
      <canvas id="vol" height="90"></canvas>
    </div>

    <!-- Lama Section -->
    <div class="card">
      <h3>تحلیل هوشمند</h3>
      <div class="row">
        <input type="text" id="lama-query" placeholder="سوال خود را بپرسید (مثال: تحلیل روند سهام)" style="flex:1">
        <button id="btn-lama">ارسال سوال</button>
      </div>
      <pre id="lama-response" class="muted"></pre>
    </div>

    <!-- DeepSeek Section -->
    <div class="card">
      <h3>پیش‌بینی قیمت</h3>
      <div class="row">
        <input type="text" id="deepseek-symbol" placeholder="نماد (مثال: وبملت)" style="flex:1">
        <button id="btn-deepseek">پیش‌بینی</button>
      </div>
      <pre id="deepseek-response" class="muted"></pre>
    </div>

    <!-- STT Section (Speech-to-Text) -->
    <div class="card">
      <h3>تحلیل صوتی</h3>
      <div class="row">
        <button id="btn-record">شروع ضبط</button>
        <button id="btn-stop-record">توقف و تحلیل</button>
      </div>
      <p id="stt-status" class="muted"></p>
      <pre id="stt-transcription" class="muted"></pre>
      <pre id="stt-lama-reply" class="muted"></pre>
    </div>

    <!-- TTS Section (Text-to-Speech) -->
    <div class="card">
      <h3>متن به صوت</h3>
      <div class="row">
        <input type="text" id="tts-text" placeholder="متن را وارد کنید..." style="flex:1">
        <button id="btn-tts">تبدیل و پخش</button>
      </div>
    </div>
  </div>

<script>
let series = null;
let ind = null;
let chart = null;
let volChart = null;
let mediaRecorder = null;
let audioChunks = [];

function qs(id) { return document.getElementById(id); }
function mode() { return document.querySelector('input[name="mode"]:checked').value; }
function log(obj) { 
  const out = qs('out') || document.createElement('pre');
  out.id = 'out';
  out.className = 'muted';
  out.textContent = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2);
  document.querySelector('.container').appendChild(out);
}

async function fetchReal() {
  const s = qs('sym').value.trim();
  if (!s) { alert('لطفاً نماد را وارد کنید'); return; }
  const p = qs('period').value;
  const i = qs('interval').value;
  const res = await fetch(`/prices?symbol=${encodeURIComponent(s)}&period=${p}&interval=${i}`);
  const data = await res.json();
  if (!res.ok) { log(data); return; }
  series = data;
  ind = null;
  log({ info: "real-data", last: { date: series.dates.at(-1), close: series.close.at(-1) } });
  draw();
}

async function fetchTSE() {
  const s = qs('sym_tse').value.trim();
  if (!s) { alert('نماد فارسی را وارد کنید (مثلاً: وبملت)'); return; }
  const url = `/tse/prices?symbol=${encodeURIComponent(s)}&adjust=true&include_jdate=true`;
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) { log(data); return; }
  series = data;
  ind = null;
  log({ info: "tse-data", last: { date: series.dates.at(-1), close: series.close.at(-1) } });
  draw();
}

async function computeIndicators() {
  if (!series) { alert('ابتدا داده را از Yahoo یا بورس تهران دریافت کنید.'); return; }
  const payload = { 
    close: series.close, 
    sma_periods: [20, 50], 
    ema_periods: [12, 26], 
    rsi_period: 14, 
    macd_fast: 12, 
    macd_slow: 26, 
    macd_signal: 9, 
    bb_period: 20, 
    bb_std: 2 
  };
  const res = await fetch('/indicators', { 
    method: 'POST', 
    headers: { 'Content-Type': 'application/json' }, 
    body: JSON.stringify(payload) 
  });
  ind = await res.json();
  log({ info: "indicators", sample: { rsi_last: ind.rsi?.at(-1) } });
  draw();
}

function toFinancialPoints(dates, ohlc) {
  if (!ohlc || !ohlc.open || !ohlc.high || !ohlc.low || !ohlc.close) return null;
  const n = Math.min(dates.length, ohlc.close.length);
  const pts = [];
  for (let i = 0; i < n; i++) {
    const o = Number(ohlc.open[i]);
    const h = Number(ohlc.high[i]);
    const l = Number(ohlc.low[i]);
    const c = Number(ohlc.close[i]);
    if ([o, h, l, c].some(x => Number.isNaN(x))) continue;
    pts.push({ x: i + 1, o, h, l, c });
  }
  return pts.length ? pts : null;
}

function draw() {
  if (!series) return;
  const ctx = qs('ch').getContext('2d');
  const vctx = qs('vol').getContext('2d');
  if (chart) chart.destroy();
  if (volChart) volChart.destroy();

  const labels = series.close.map((_, i) => i + 1);
  const ds = [];

  if (mode() === 'candle') {
    const fin = toFinancialPoints(series.dates, series.ohlc);
    if (fin) { ds.push({ label: 'Candlestick', data: fin, type: 'candlestick' }); }
    else { ds.push({ label: 'Close', data: series.close, borderWidth: 1, pointRadius: 0 }); }
  } else {
    ds.push({ label: 'Close', data: series.close, borderWidth: 1, pointRadius: 0 });
  }

  if (ind) {
    if (ind.sma && ind.sma['20']) ds.push({ label: 'SMA20', data: ind.sma['20'], borderWidth: 1, pointRadius: 0 });
    if (ind.sma && ind.sma['50']) ds.push({ label: 'SMA50', data: ind.sma['50'], borderWidth: 1, pointRadius: 0 });
    if (mode() !== 'candle') {
      if (ind.ema && ind.ema['12']) ds.push({ label: 'EMA12', data: ind.ema['12'], borderWidth: 1, pointRadius: 0 });
      if (ind.ema && ind.ema['26']) ds.push({ label: 'EMA26', data: ind.ema['26'], borderWidth: 1, pointRadius: 0 });
    }
    if (ind.bb) {
      if (ind.bb.upper) ds.push({ label: 'BB Upper', data: ind.bb.upper, borderWidth: 1, pointRadius: 0 });
      if (ind.bb.middle) ds.push({ label: 'BB Middle', data: ind.bb.middle, borderWidth: 1, pointRadius: 0 });
      if (ind.bb.lower) ds.push({ label: 'BB Lower', data: ind.bb.lower, borderWidth: 1, pointRadius: 0 });
    }
  }

  chart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: ds },
    options: {
      responsive: true,
      animation: false,
      plugins: { legend: { display: true }, decimation: { enabled: true, algorithm: 'lttb', samples: 300 } },
      elements: { line: { tension: 0.2 } },
      scales: { x: { ticks: { display: false } } }
    }
  });

  if (series.volume && Array.isArray(series.volume)) {
    volChart = new Chart(vctx, {
      type: 'bar',
      data: { labels, datasets: [{ label: 'Volume', data: series.volume }] },
      options: { responsive: true, animation: false, scales: { x: { ticks: { display: false } } } }
    });
  }
}

qs('btn-lama').onclick = async function () {
  const query = qs('lama-query').value;
  if (!query) { alert('لطفاً یک سوال وارد کنید.'); return; }
  const payload = { prompt: query };
  if (ind) { payload.indicators = ind; }
  try {
    const response = await fetch('/lama_predict', {
      method: 'POST', 
      headers: { 'Content-Type': 'application/json' }, 
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    qs('lama-response').textContent = data.response || 'پاسخی دریافت نشد.';
  } catch (e) {
    qs('lama-response').textContent = 'خطا در دریافت پاسخ: ' + e.message;
  }
};

qs('btn-deepseek').onclick = async function () {
  const symbol = qs('deepseek-symbol').value || 'وبملت';
  try {
    const response = await fetch(`/deepseek_predict?symbol=${encodeURIComponent(symbol)}`);
    const data = await response.json();
    if (data.prediction !== undefined) {
      const p = Number(data.prediction).toFixed(2);
      const s = Number(data.uncertainty_std).toFixed(2);
      qs('deepseek-response').textContent = `قیمت پیش‌بینی‌شده: ${p} (انحراف معیار: ${s})`;
    } else {
      qs('deepseek-response').textContent = data.detail || 'پیش‌بینی ناموفق بود';
    }
  } catch (e) {
    qs('deepseek-response').textContent = 'خطا در پیش‌بینی: ' + e.message;
  }
};

qs('btn-real').onclick = fetchReal;
qs('btn-tse').onclick = fetchTSE;
qs('btn-ind').onclick = computeIndicators;

qs('btn-tts').onclick = async () => {
  const text = qs('tts-text').value;
  if (!text) { alert('لطفاً متن را وارد کنید.'); return; }
  try {
    const formData = new FormData();
    formData.append('text', text);
    const res = await fetch('/api/tts', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    if (data.audio_file) {
      const audioUrl = `/api/download?path=${encodeURIComponent(data.audio_file)}`;
      const audio = new Audio(audioUrl);
      audio.play().catch(e => alert('خطا در پخش صوت: ' + e.message));
    } else {
      throw new Error('No audio file returned');
    }
  } catch (e) {
    alert('خطا در تبدیل به صوت: ' + e.message);
  }
};

qs('btn-record').onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();
    qs('stt-status').textContent = 'در حال ضبط...';
    audioChunks = [];
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = sendSTT;
  } catch (e) {
    qs('stt-status').textContent = 'خطا در دسترسی به میکروفون: ' + e.message;
  }
};

qs('btn-stop-record').onclick = () => {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    qs('stt-status').textContent = 'ضبط متوقف شد. در حال تحلیل...';
  } else {
    qs('stt-status').textContent = 'هیچ ضبطی در حال انجام نیست.';
  }
};

async function sendSTT() {
  if (audioChunks.length === 0) {
    qs('stt-status').textContent = 'هیچ صوتی ضبط نشده است.';
    return;
  }

  const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
  const formData = new FormData();
  formData.append('file', audioBlob, 'audio.wav');

  try {
    const res = await fetch('/api/stt', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    qs('stt-transcription').textContent = 'متن استخراج‌شده: ' + data.transcription;
    qs('stt-lama-reply').textContent = 'پاسخ تحلیل: ' + data.llama_text_reply;
    if (data.audio_file) {
      const audioUrl = `/api/download?path=${encodeURIComponent(data.audio_file)}`;
      const audio = new Audio(audioUrl);
      audio.play().catch(e => qs('stt-status').textContent = 'خطا در پخش صوت: ' + e.message);
      qs('stt-status').textContent = 'تحلیل و پخش کامل شد.';
    } else {
      qs('stt-status').textContent = 'فایل صوتی تولید نشد.';
    }
  } catch (e) {
    qs('stt-status').textContent = 'خطا: ' + e.message;
  }
}
</script>
</body>
</html>
    """
    return HTMLResponse(content=html, status_code=200)

# ================================
# Run with uvicorn
# ================================
if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "uvicorn missing. Inside venv run: pip install uvicorn fastapi pandas numpy yfinance python-multipart pytse-client scikit-learn pyttsx3 whisper"
        )
    module_path = f"{Path(__file__).stem}:app"
    uvicorn.run(module_path, host="127.0.0.1", port=8082, reload=True)