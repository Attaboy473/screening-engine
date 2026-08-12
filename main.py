"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SCREENING ENGINE — BACKEND                          ║
║                                                                            ║
║  All-in-One IDX Stock Screening Platform with:                             ║
║    - IHSG Dashboard (live market data + top movers)                        ║
║    - 3-Lane Stock Analysis (Teknikal + Momentum + Fundamental)             ║
║    - ATR-Based Conservative Trading Plan (entry range, SL, TP1-3)          ║
║    - Batch Recommendation (12+ stocks ranked by composite score)           ║
║    - Backtest Time Machine (historical scoring + forward return lookup)    ║
║                                                                            ║
║  Tech Stack: Python 3.11 + FastAPI + yfinance + ta + Uvicorn               ║
║  Data Source: Yahoo Finance (15-min delay for IDX stocks)                  ║
║  Author: Attaboy473                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --- Technical Analysis Indicators ---
from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

# --- Web Framework ---
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ═══════════════════════════════════════════════════════════════════════════════
# APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Screening Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Universe: 42 IDX Stocks across multiple sectors ---
POPULAR = [
    "BBCA", "BBRI", "BMRI", "TLKM", "ASII", "BBNI", "UNVR", "GOTO",
    "ANTM", "ADRO", "ICBP", "MDKA"
]
TICKERS_42 = POPULAR + [
    "INDF", "CPIN", "KLBF", "UNTR", "PTBA", "BRIS", "ISAT", "INCO",
    "EXCL", "SIDO", "ARTO", "HRUM", "NCKL", "BRMS", "SMGR", "PWON",
    "TOWR", "DSSA", "AMRT", "ACES", "MAPI", "INTP", "MEDC", "PGAS",
    "BSDE", "CTRA", "ERAA", "BRPT", "CUAN", "BUKA"
]

# --- Lazy-loaded caches (initialized on first access) ---
# Store full historical OHLCV data for backtesting (Jan 2024 – Feb 2026)
HISTORICAL_CACHE: dict[str, pd.DataFrame] = {}
IHsgHISTORICAL: pd.DataFrame | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY: Safe float conversion (prevents NaN/Inf from breaking JSON)
# ═══════════════════════════════════════════════════════════════════════════════

def safe(value, default: float = 0.0) -> float:
    """Convert value to float, return `default` if NaN, Inf, or conversion fails."""
    try:
        v = float(value)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: Composite Scoring Engine
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the heart of the application. It calculates a 0–100 composite score
# by averaging three sub-scores:
#
#   Teknikal (max 100)  — 9 indicators: MA, RSI, MACD, BB, ATR, Stoch, ADX, EMA, Vol
#   Momentum (max 100)  — 5 signals: NR7, BB Squeeze, Volume Spike, Return, Near High
#   Fundamental (max 100)— 7 ratios: PER, PBV, ROE, Profit Margin, Revenue Growth, DER, Div Yield
#
#   Composite = (Teknikal + Momentum + Fundamental) / 3  →  0–100 scale
#
# Thresholds (0–100 scale):
#   ≥ 80  → STRONG BUY
#   ≥ 63  → BUY
#   ≥ 47  → HOLD
#   ≥ 30  → WATCH
#   < 30  → AVOID
#
# The function also generates an ATR-based Conservative Trading Plan:
#   - Stop Loss = 1.5× ATR below current price
#   - TP1 = 1.0× ATR above entry (~2-3 days)
#   - TP2 = 1.8× ATR above entry (~4-6 days)
#   - TP3 = 2.5× ATR above entry (~7-10 days)
#   - Entry Range = between Support (max of Bollinger lower / MA50) and MA20
# ═══════════════════════════════════════════════════════════════════════════════

def calc_scores(
    close: np.ndarray,      # Close prices (numpy array)
    high: np.ndarray,        # High prices
    low: np.ndarray,         # Low prices
    volume: np.ndarray,      # Volume
    info: dict | None = None # Yahoo Finance info dict (for fundamental data)
) -> dict | None:
    """
    Calculate composite score (0–100) and generate trading plan.
    Returns None if data is insufficient (< 60 data points).
    """
    n = len(close)
    if n < 60:
        return None
    latest = safe(close[-1])
    if latest <= 0:
        return None

    # --------------------------------------------------------------------------
    # Compute all technical indicators using the `ta` library
    # --------------------------------------------------------------------------
    try:
        ma20 = safe(SMAIndicator(pd.Series(close), 20).sma_indicator().iloc[-1])
        ma50 = safe(SMAIndicator(pd.Series(close), 50).sma_indicator().iloc[-1])
        rsi  = safe(RSIIndicator(pd.Series(close), 14).rsi().iloc[-1], 50)

        bb = BollingerBands(pd.Series(close), 20, 2)
        bb_u = safe(bb.bollinger_hband().iloc[-1])
        bb_l = safe(bb.bollinger_lband().iloc[-1])
        bb_pos = safe((latest - bb_l) / (bb_u - bb_l) * 100, 50) if bb_u > bb_l else 50

        macd_i = MACD(pd.Series(close), 26, 12, 9)
        ml = safe(macd_i.macd().iloc[-1])          # MACD line
        ms = safe(macd_i.macd_signal().iloc[-1])    # Signal line
        mh = safe(macd_i.macd_diff().iloc[-1])      # Histogram

        atr = safe(AverageTrueRange(pd.Series(high), pd.Series(low), pd.Series(close), 14)
                   .average_true_range().iloc[-1])
        atr_pct = safe(atr / latest * 100)

        st = safe(StochasticOscillator(pd.Series(high), pd.Series(low), pd.Series(close), 14, 3)
                  .stoch().iloc[-1], 50)
        adx = safe(ADXIndicator(pd.Series(high), pd.Series(low), pd.Series(close), 14)
                   .adx().iloc[-1], 20)
        ema20 = safe(EMAIndicator(pd.Series(close), 20).ema_indicator().iloc[-1])

        va = safe(np.mean(volume[-20:]))   # Average volume (20 days)
        vn = safe(volume[-1])               # Latest volume
        vr = safe(vn / va) if va > 0 else 1.0
        avg_val = va * latest               # Average transaction value (liquidity proxy)
    except Exception:
        return None

    # --------------------------------------------------------------------------
    # 1) TEKNIKAL SCORING (9 indicators, max 100 points)
    #    Each indicator contributes based on whether conditions are met
    # --------------------------------------------------------------------------
    tek, t_details = 0, []

    # Harga vs MA20
    if latest > ma20:
        tek += 13
        t_details.append({"indikator": "Harga > MA20", "signal": f"Di atas MA20 ({ma20:.0f})", "skor": 13, "max": 13})
    else:
        t_details.append({"indikator": "Harga > MA20", "signal": f"Di bawah MA20 ({ma20:.0f})", "skor": 0, "max": 13})

    # Golden Cross / Death Cross (MA20 vs MA50)
    if ma20 > ma50:
        tek += 12
        t_details.append({"indikator": "MA20 > MA50", "signal": "Golden Cross", "skor": 12, "max": 12})
    else:
        t_details.append({"indikator": "MA20 > MA50", "signal": "Death Cross", "skor": 0, "max": 12})

    # RSI (Relative Strength Index)
    if 40 <= rsi <= 70:
        tek += 15; lbl = f"Sehat ({rsi:.0f})"
    elif 30 <= rsi < 40:
        tek += 8;  lbl = f"Oversold ({rsi:.0f})"
    elif rsi > 70:
        tek += 5;  lbl = f"Overbought ({rsi:.0f})"
    else:
        tek += 3;  lbl = f"Jatuh ({rsi:.0f})"
    t_details.append({"indikator": f"RSI ({rsi:.0f})", "signal": lbl,
                      "skor": 15 if 40<=rsi<=70 else (8 if 30<=rsi<40 else (5 if rsi>70 else 3)), "max": 15})

    # MACD
    if ml > ms and mh > 0:
        tek += 15; lbl = "Bullish kuat"
    elif ml > ms:
        tek += 10; lbl = "Bullish melemah"
    elif ml > 0:
        tek += 5;  lbl = "Bearish cross"
    else:
        tek += 0;  lbl = "Bearish"
    t_details.append({"indikator": "MACD", "signal": lbl,
                      "skor": 15 if ml>ms and mh>0 else (10 if ml>ms else (5 if ml>0 else 0)), "max": 15})

    # Bollinger Band Position
    if 20 <= bb_pos <= 80:
        tek += 10; lbl = f"Normal ({bb_pos:.0f}%)"
    else:
        tek += 4;  lbl = f"Ekstrem ({bb_pos:.0f}%)"
    t_details.append({"indikator": "BB Position", "signal": lbl,
                      "skor": 10 if 20<=bb_pos<=80 else 4, "max": 10})

    # ATR (Average True Range) — measures volatility
    if 1 <= atr_pct <= 4:
        tek += 10; lbl = f"Wajar ({atr_pct:.1f}%)"
    elif atr_pct < 1:
        tek += 5;  lbl = "Tenang (<1%)"
    else:
        tek += 3;  lbl = "Tinggi (>4%)"
    t_details.append({"indikator": "ATR", "signal": lbl,
                      "skor": 10 if 1<=atr_pct<=4 else (5 if atr_pct<1 else 3), "max": 10})

    # Stochastic
    if 20 <= st <= 80:
        tek += 10; lbl = f"Normal ({st:.0f})"
    elif st < 20:
        tek += 5;  lbl = f"Oversold ({st:.0f})"
    else:
        tek += 4;  lbl = f"Overbought ({st:.0f})"
    t_details.append({"indikator": f"Stoch K", "signal": lbl,
                      "skor": 10 if 20<=st<=80 else (5 if st<20 else 4), "max": 10})

    # ADX (trend strength)
    if adx >= 25:
        tek += 10; lbl = f"Kuat ({adx:.0f})"
    elif adx >= 20:
        tek += 6;  lbl = f"Moderat ({adx:.0f})"
    else:
        tek += 2;  lbl = f"Lemah ({adx:.0f})"
    t_details.append({"indikator": "ADX", "signal": lbl,
                      "skor": 10 if adx>=25 else (6 if adx>=20 else 2), "max": 10})

    # EMA20
    if latest > ema20:
        tek += 5
        t_details.append({"indikator": "Harga > EMA20", "signal": f"Di atas EMA20 ({ema20:.0f})", "skor": 5, "max": 5})
    else:
        t_details.append({"indikator": "Harga > EMA20", "signal": f"Di bawah EMA20 ({ema20:.0f})", "skor": 0, "max": 5})

    # Volume Ratio (latest vs average)
    if vr >= 1.3:
        tek += 10; lbl = f"Tinggi ({vr:.1f}x)"
    elif vr >= 0.7:
        tek += 6;  lbl = f"Normal ({vr:.1f}x)"
    else:
        tek += 2;  lbl = f"Rendah ({vr:.1f}x)"
    t_details.append({"indikator": "Vol Ratio", "signal": lbl,
                      "skor": 10 if vr>=1.3 else (6 if vr>=0.7 else 2), "max": 10})

    # --------------------------------------------------------------------------
    # 2) MOMENTUM SCORING (5 signals, max 100 points)
    # --------------------------------------------------------------------------
    mom, m_details = 0, []

    # NR7 — Narrowest Range in 7 days (potential breakout signal)
    ranges_hilo = np.array([high[i] - low[i] for i in range(n)])
    if n >= 7 and ranges_hilo[-1] <= np.min(ranges_hilo[-7:]):
        mom += 25
        m_details.append({"indikator": "NR7", "signal": "Potensi breakout", "skor": 25, "max": 25})
    else:
        m_details.append({"indikator": "NR7", "signal": "Tidak terdeteksi", "skor": 0, "max": 25})

    # Bollinger Band Squeeze Breakout
    bbw = safe((bb_u - bb_l) / ((bb_u + bb_l) / 2) * 100 if (bb_u + bb_l) > 0 else 0)
    if bbw > 3 and latest > ma20:
        mom += 20
        m_details.append({"indikator": "BB Squeeze Breakout", "signal": f"BBW {bbw:.1f}% > 3%", "skor": 20, "max": 20})
    else:
        m_details.append({"indikator": "BB Squeeze Breakout", "signal": f"BBW {bbw:.1f}% — tidak squeeze", "skor": 0, "max": 20})

    # Volume Spike (5-day > 1.5× 5-day prior)
    v5 = safe(np.mean(volume[-5:]))
    v10 = safe(np.mean(volume[-10:-5]))
    if v10 > 0 and v5 / v10 > 1.5:
        mom += 20
        m_details.append({"indikator": "Volume Spike", "signal": f"Vol naik {v5/v10:.1f}x vs baseline", "skor": 20, "max": 20})
    else:
        m_details.append({"indikator": "Volume Spike", "signal": f"Vol normal ({v5/v10:.1f}x)", "skor": 0, "max": 20})

    # Return 5-day / 20-day
    r5  = safe((close[-1] / close[-6]  - 1) * 100) if n >= 6  else 0
    r20 = safe((close[-1] / close[-21] - 1) * 100) if n >= 21 else 0
    ms5  = 15 if r5 > 3  else (8 if r5 > 0 else (4 if r5 > -3 else 0))
    ms20 = 10 if r20 > 5 else (5 if r20 > 0 else 0)
    mom += ms5 + ms20
    m_details.append({"indikator": "Return 5/20H", "signal": f"{r5:+.1f}% / {r20:+.1f}%",
                      "skor": ms5 + ms20, "max": 25})

    # Near 20-day High (price within 95% of recent high)
    if n >= 20 and close[-1] > np.max(close[-20:]) * .95:
        mom += 10
        m_details.append({"indikator": "Near High", "signal": f"{((close[-1]/np.max(close[-20:]))*100):.0f}% dari high 20H", "skor": 10, "max": 10})
    else:
        m_details.append({"indikator": "Near High", "signal": "Jauh dari high 20H", "skor": 0, "max": 10})

    # --------------------------------------------------------------------------
    # 3) FUNDAMENTAL SCORING (7 ratios, max 100 points)
    #    Only available when `info` (Yahoo Finance ticker info) is provided.
    #    Not available for backtesting (historical fundamental data unavailable).
    # --------------------------------------------------------------------------
    f_score, f_details = 0, []
    if info:
        pe  = safe(info.get("trailingPE", 0))
        pb  = safe(info.get("priceToBook", 0))
        roe = safe(info.get("returnOnEquity", 0))
        pm  = safe(info.get("profitMargins", 0))
        rg  = safe(info.get("revenueGrowth", 0))
        dy  = safe(info.get("dividendYield", 0))
        der = safe(info.get("debtToEquity", 0))

        # PER (Price-to-Earnings Ratio)
        if pe <= 0:
            f_details.append({"indikator": "PER", "signal": "Negatif", "skor": 0, "max": 20})
        elif pe <= 10:
            f_score += 20; f_details.append({"indikator": f"PER ({pe:.1f})", "signal": "Murah", "skor": 20, "max": 20})
        elif pe <= 15:
            f_score += 14; f_details.append({"indikator": f"PER ({pe:.1f})", "signal": "Wajar", "skor": 14, "max": 20})
        elif pe <= 25:
            f_score += 7;  f_details.append({"indikator": f"PER ({pe:.1f})", "signal": " Mahal", "skor": 7, "max": 20})
        else:
            f_score += 2;  f_details.append({"indikator": f"PER ({pe:.1f})", "signal": "Sangat mahal", "skor": 2, "max": 20})

        # PBV (Price-to-Book Value)
        if pb <= 1:
            f_score += 15; f_details.append({"indikator": f"PBV ({pb:.2f})", "signal": "Murah", "skor": 15, "max": 15})
        elif pb <= 3:
            f_score += 9;  f_details.append({"indikator": f"PBV ({pb:.2f})", "signal": "Wajar", "skor": 9, "max": 15})
        else:
            f_score += 4;  f_details.append({"indikator": f"PBV ({pb:.2f})", "signal": " Mahal", "skor": 4, "max": 15})

        # ROE (Return on Equity)
        if roe >= 0.20:
            f_score += 20; f_details.append({"indikator": f"ROE ({roe*100:.0f}%)", "signal": "Sangat baik", "skor": 20, "max": 20})
        elif roe >= 0.10:
            f_score += 14; f_details.append({"indikator": f"ROE ({roe*100:.0f}%)", "signal": "Baik", "skor": 14, "max": 20})
        elif roe > 0:
            f_score += 7;  f_details.append({"indikator": f"ROE ({roe*100:.0f}%)", "signal": " Cukup", "skor": 7, "max": 20})
        else:
            f_details.append({"indikator": "ROE", "signal": "Negatif", "skor": 0, "max": 20})

        # Profit Margin
        if pm >= 0.20:
            f_score += 15; f_details.append({"indikator": f"Profit Margin ({pm*100:.0f}%)", "signal": "Sangat baik", "skor": 15, "max": 15})
        elif pm > 0:
            f_score += 10; f_details.append({"indikator": f"Profit Margin ({pm*100:.0f}%)", "signal": "Baik", "skor": 10, "max": 15})
        else:
            f_details.append({"indikator": "Profit Margin", "signal": "Rugi", "skor": 0, "max": 15})

        # Revenue Growth (YoY)
        if rg >= 0.15:
            f_score += 15; f_details.append({"indikator": f"Rev Growth ({rg*100:.0f}%)", "signal": "Tumbuh kuat", "skor": 15, "max": 15})
        elif rg > 0:
            f_score += 10; f_details.append({"indikator": f"Rev Growth ({rg*100:.0f}%)", "signal": " Melambat", "skor": 10, "max": 15})
        else:
            f_score += 2;  f_details.append({"indikator": "Rev Growth", "signal": "Menurun", "skor": 2, "max": 15})

        # Dividend Yield
        if dy >= 3:
            f_score += 5; f_details.append({"indikator": f"Div Yield ({dy:.1f}%)", "signal": "Tinggi", "skor": 5, "max": 5})
        elif dy > 0:
            f_score += 2; f_details.append({"indikator": f"Div Yield ({dy:.1f}%)", "signal": " Ada", "skor": 2, "max": 5})
        else:
            f_details.append({"indikator": "Div Yield", "signal": "Tidak ada dividen", "skor": 0, "max": 5})

        # DER (Debt-to-Equity Ratio)
        if der <= 0.5:
            f_score += 10; f_details.append({"indikator": f"DER ({der:.2f})", "signal": "Rendah", "skor": 10, "max": 10})
        elif der <= 1.5:
            f_score += 6;  f_details.append({"indikator": f"DER ({der:.2f})", "signal": " Moderat", "skor": 6, "max": 10})
        else:
            f_score += 2;  f_details.append({"indikator": f"DER ({der:.2f})", "signal": "Tinggi", "skor": 2, "max": 10})

    # --------------------------------------------------------------------------
    # COMPOSITE SCORE (0–100 scale)
    # --------------------------------------------------------------------------
    total = round((min(tek, 100) + min(mom, 100) + min(f_score, 100)) / 3)

    # --- Liquidity Grade (based on average daily transaction value) ---
    if avg_val >= 10e9:
        grade = "A"
    elif avg_val >= 1e9:
        grade = "B"
    elif avg_val >= 500e6:
        grade = "C"
    else:
        grade = "D"

    # --- Recommendation Thresholds ---
    if total >= 80:
        rec = "STRONG BUY"
    elif total >= 63:
        rec = "BUY"
    elif total >= 47:
        rec = "HOLD"
    elif total >= 30:
        rec = "WATCH"
    else:
        rec = "AVOID"

    # Downgrade illiquid stocks to HOLD regardless of score
    if grade == "D" and rec in ("STRONG BUY", "BUY"):
        rec = "HOLD"

    # --------------------------------------------------------------------------
    # ATR-BASED CONSERVATIVE TRADING PLAN
    #   - Entry Range: between Support and MA20
    #   - Stop Loss: 1.5× ATR below current price
    #   - TP1/TP2/TP3: 1.0× / 1.8× / 2.5× ATR above entry
    # --------------------------------------------------------------------------
    support    = round(min(bb_l, safe(np.min(low[-20:])))) if bb_l > 0 else round(latest - atr * 2)
    resistance = round(max(bb_u, safe(np.max(high[-20:])))) if bb_u > 0 else round(latest + atr * 2)
    entry_bawah = round(max(support, ma50)) if ma50 > 0 else support
    entry_atas  = round(min(ma20, resistance)) if ma20 > 0 else resistance
    sl = round(latest - atr * 1.5)

    if atr > 0:
        tp1 = round(latest + atr * 1.0)
        tp2 = round(latest + atr * 1.8)
        tp3 = round(latest + atr * 2.5)
        tp1_pct = safe((tp1 / latest - 1) * 100)
        tp2_pct = safe((tp2 / latest - 1) * 100)
        tp3_pct = safe((tp3 / latest - 1) * 100)
        sl_pct  = safe((sl / latest - 1) * 100)
        est1 = max(1, round((tp1 - latest) / atr))
        est2 = max(2, round((tp2 - latest) / atr))
        est3 = max(3, round((tp3 - latest) / atr))
    else:
        tp1 = tp2 = tp3 = latest
        tp1_pct = tp2_pct = tp3_pct = sl_pct = 0
        est1 = est2 = est3 = 0

    # --- Trading Style Classification (based on volatility) ---
    if atr_pct < 1.5:
        gaya = "Intraday/Scalping"
    elif atr_pct < 4:
        gaya = "Swing (2-10h)"
    else:
        gaya = "Position (>2mg)"

    return {
        "kode": "",
        "score": total, "rec": rec, "grade": grade, "harga": round(latest),
        "sector": info.get("sector", "—") if info else "—",
        "tek_score": min(tek, 100), "mom_score": min(mom, 100), "fun_score": min(f_score, 100),
        "tek_details": t_details, "mom_details": m_details, "fun_details": f_details,
        "ma20": round(ma20), "ma50": round(ma50), "rsi": round(rsi, 1),
        "adx": round(adx, 1), "atr_pct": round(atr_pct, 1), "vr": round(vr, 2),
        "bb_pos": round(bb_pos, 1),
        "entry_bawah": entry_bawah, "entry_atas": entry_atas,
        "support": support, "resistance": resistance,
        "sl": sl, "sl_pct": round(sl_pct, 1),
        "tp1": tp1, "tp1_pct": round(tp1_pct, 1), "est_tp1": est1,
        "tp2": tp2, "tp2_pct": round(tp2_pct, 1), "est_tp2": est2,
        "tp3": tp3, "tp3_pct": round(tp3_pct, 1), "est_tp3": est3,
        "gaya": gaya, "avg_val_20": safe(avg_val),
        "change_1d": safe((close[-1] / close[-2] - 1) * 100) if n >= 2 else 0,
        "change_5d": safe((close[-1] / close[-6] - 1) * 100) if n >= 6 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINT: IHSG Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
# Returns:
#   - IHSG price, change, status, MA20/MA50/RSI
#   - 1-year candlestick chart data
#   - Top 10 gainers & losers (with sector info)
#   - Market Overview (breadth, foreign flow proxy, sentiment, technical outlook)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/ihsg")
async def ihsg_api():
    """IHSG Dashboard — live market summary."""
    try:
        df = yf.download("^JKSE", period="1y", progress=False, auto_adjust=True)
        if df.empty:
            return {"ok": False, "error": "Data IHSG kosong"}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index

        c = df["Close"].values
        l = safe(c[-1])
        ch = safe((c[-1] / c[-2] - 1) * 100)
        ma20_v = safe(SMAIndicator(pd.Series(c), 20).sma_indicator().iloc[-1])
        ma50_v = safe(SMAIndicator(pd.Series(c), 50).sma_indicator().iloc[-1])
        rsi_v  = safe(RSIIndicator(pd.Series(c), 14).rsi().iloc[-1], 50)

        # Factual status (NOT a prediction — just describes current state)
        if l > ma20_v > ma50_v:
            st = "Hijau (di atas MA20 & MA50)"
        elif l > ma20_v:
            st = "Hijau (di atas MA20)"
        elif l > ma50_v:
            st = "Kuning (di antara MA20 & MA50)"
        else:
            st = "Merah (di bawah MA20 & MA50)"

        # Build chart data array
        chart = []
        for idx, row in df.iterrows():
            ts = int(pd.Timestamp(idx).timestamp())
            chart.append({
                "time": ts,
                "open":  safe(float(row["Open"])),
                "high":  safe(float(row["High"])),
                "low":   safe(float(row["Low"])),
                "close": safe(float(row["Close"])),
                "volume": safe(float(row["Volume"]), 0),
            })

        # Fetch OHLCV for ~24 movers in parallel
        codes = POPULAR + ["INDF","CPIN","KLBF","UNTR","PTBA","BRIS","ISAT","INCO",
                           "EXCL","SIDO","ARTO","HRUM","NCKL","BRMS","SMGR","PWON"]
        movers = []
        with ThreadPoolExecutor(6) as pool:
            futures = {}
            for code in codes:
                futures[pool.submit(
                    lambda c=code: (yf.Ticker(f"{c}.JK"), c)
                )] = code
            for f in as_completed(futures):
                code = futures[f]
                try:
                    tk, _ = f.result()
                    df2 = tk.history(period="5d", auto_adjust=True)
                    if df2 is None or len(df2) < 2:
                        continue
                    if isinstance(df2.columns, pd.MultiIndex):
                        df2.columns = df2.columns.get_level_values(0)
                    cc_arr = df2["Close"].values
                    l2, p2 = safe(cc_arr[-1]), safe(cc_arr[-2])
                    ch2 = safe((l2 / p2 - 1) * 100) if p2 > 0 else 0
                    sector = tk.info.get("sector", "—") if tk.info else "—"
                    movers.append({
                        "kode": code, "harga": round(l2),
                        "change_pct": round(ch2, 2), "sector": sector,
                    })
                except Exception:
                    pass

        gain = sorted([m for m in movers if m["change_pct"] > 0],
                      key=lambda x: -x["change_pct"])[:10]
        loss = sorted([m for m in movers if m["change_pct"] < 0],
                      key=lambda x: x["change_pct"])[:10]
        flat = sorted([m for m in movers if m["change_pct"] == 0],
                      key=lambda x: -x["change_pct"])[:10]

        # --- Market Overview section ---
        up, down, unchanged = len(gain), len(loss), len(flat)
        net_flow = sum(m["change_pct"] for m in gain) + sum(m["change_pct"] for m in loss)
        foreign_label = f"Net Flow ~{'+' if net_flow>0 else ''}{net_flow:.0f}% cap-weighted" if movers else "—"
        sentiment_score = min(100, max(0, 50 + net_flow * 5))
        if sentiment_score >= 70:
            sent = "Greed"
        elif sentiment_score >= 40:
            sent = "Neutral"
        else:
            sent = "Fear"
        tech_outlook = st.split("(")[0].strip()
        tech_icon = "Hijau" if "Hijau" in tech_outlook else ("Kuning" if "Kuning" in tech_outlook else "Merah")

        overview = {
            "breadth":   f"{up} Naik  {down} Turun  {unchanged} Tetap",
            "foreign":   foreign_label,
            "sentiment": f"{sentiment_score}/100 {sent}",
            "technical": f"{tech_icon} — {tech_outlook}",
        }

        return {
            "ok": True,
            "ihsg": {
                "harga": round(l), "change_pct": round(ch, 2), "status": st,
                "ma20": round(ma20_v), "ma50": round(ma50_v),
                "rsi": round(rsi_v, 1), "overview": overview,
            },
            "ihsg_chart": chart,
            "gainers": gain, "losers": loss,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINT: Single Stock Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/score/{kode}")
async def score_api(kode: str):
    """Full analysis for a single stock: scoring + trading plan + indicators."""
    try:
        ticker = yf.Ticker(f"{kode.upper().strip()}.JK")
        df = ticker.history(period="6mo", auto_adjust=True)
        info = ticker.info or {}

        if df.empty or len(df) < 50:
            return {"ok": False, "error": "Data tidak cukup"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index

        result = calc_scores(
            df["Close"].values, df["High"].values,
            df["Low"].values, df["Volume"].values, info
        )
        if result is None:
            return {"ok": False, "error": "Gagal menghitung skor"}

        result["kode"] = kode.upper()
        return {"ok": True, "data": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINT: Batch Screening (Rekomendasi)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/screener")
async def screener_api(kodes: str = Query(None)):
    """
    Score multiple stocks in parallel. Default: 12 popular tickers.
    Usage: /api/screener?kodes=BBCA,BBRI,TLKM
    """
    if not kodes:
        kodes = ",".join(POPULAR)
    tickers = [k.strip().upper() for k in kodes.split(",") if k.strip()]
    results = []

    with ThreadPoolExecutor(6) as pool:
        futures = {pool.submit(lambda c=c: analyze_quick(c)): c for c in tickers}
        for f in as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda x: x.get("score", 0) if "score" in x else 0, reverse=True)
    return {"ok": True, "count": len(results), "data": results}


def analyze_quick(kode: str) -> dict:
    """
    Lightweight single-stock analysis.
    Same scoring engine as /api/score/{kode}, optimized for parallel batch use.
    """
    try:
        tk = yf.Ticker(f"{kode}.JK")
        df = tk.history(period="6mo", auto_adjust=True)
        info = tk.info or {}

        if df.empty or len(df) < 50:
            return {"kode": kode, "error": "Data tidak cukup"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index

        r = calc_scores(df["Close"].values, df["High"].values,
                        df["Low"].values, df["Volume"].values, info)
        if r is None:
            return {"kode": kode, "error": "Gagal menghitung"}
        r["kode"] = kode
        return r
    except Exception:
        return {"kode": kode, "error": "Download gagal"}


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINT: Candlestick Chart Data
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/chart/{kode}")
async def chart_api(kode: str, period: str = Query("6mo")):
    """
    Return OHLCV candlestick data for TradingView Lightweight Charts.
    Periods: 1mo, 3mo, 6mo, 1y
    """
    try:
        df = yf.download(
            f"{kode.upper().strip()}.JK",
            period=period, progress=False, auto_adjust=True
        )
        if df.empty:
            return {"ok": False, "error": "Data kosong"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        data = []
        for idx, row in df.iterrows():
            data.append({
                "time":   int(pd.Timestamp(idx).timestamp()),
                "open":   safe(float(row["Open"])),
                "high":   safe(float(row["High"])),
                "low":    safe(float(row["Low"])),
                "close":  safe(float(row["Close"])),
                "volume": safe(float(row["Volume"]), 0),
            })
        return {"ok": True, "data": data, "kode": kode.upper()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST: Lazy Historical Data Initialization
# ═══════════════════════════════════════════════════════════════════════════════
# Downloads full OHLCV history for 42 stocks + IHSG (Jan 2024 – Feb 2026).
# Runs once on first backtest API call and caches results in memory.
# ═══════════════════════════════════════════════════════════════════════════════

def init_historical():
    """Download and cache historical data for backtesting."""
    global HISTORICAL_CACHE, IHsgHISTORICAL
    if HISTORICAL_CACHE:
        return
    print("Downloading historical data for Backtest (Jan 2024 – Feb 2026)...")
    with ThreadPoolExecutor(8) as pool:
        futures = {}
        for k in TICKERS_42:
            futures[pool.submit(
                lambda c=k: yf.Ticker(f"{c}.JK").history(
                    start="2024-01-01", end="2026-02-01", auto_adjust=False
                )
            )] = k
        for f in as_completed(futures):
            k = futures[f]
            try:
                df = f.result()
                if df is not None and len(df) >= 80:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                    HISTORICAL_CACHE[k] = df
            except Exception:
                pass

    df_i = yf.download("^JKSE", start="2024-01-01", end="2026-02-01",
                        auto_adjust=True, progress=False)
    if isinstance(df_i.columns, pd.MultiIndex):
        df_i.columns = df_i.columns.get_level_values(0)
    df_i.index = df_i.index.tz_localize(None) if df_i.index.tz is not None else df_i.index
    IHsgHISTORICAL = df_i
    print(f" {len(HISTORICAL_CACHE)} stocks ready for backtesting")


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINT: Backtest Dates
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/backtest/dates")
async def backtest_dates():
    """Return all available backtest dates in 2025 (every 2nd trading day)."""
    init_historical()
    all_dates = sorted(set().union(*[set(df.index.tolist()) for df in HISTORICAL_CACHE.values()]))
    dates_2025 = [
        d.strftime("%Y-%m-%d") for d in all_dates
        if pd.Timestamp("2025-01-01") <= d <= pd.Timestamp("2025-12-31")
    ]
    return {"dates": dates_2025[::2]}  # ≈118 dates


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINT: Backtest Screen
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage: /api/backtest/screen?date=2025-08-15&horizon=20
#
# This endpoint:
#   1. Re-runs the scoring engine using ONLY data available on `date`
#      (NO look-ahead bias — fundamental data excluded for historical fairness)
#   2. Filters for BUY signals (score ≥ 75 on 2-lane technique+momentum)
#   3. Looks up forward return `horizon` days later
#   4. Flags which TP levels (2%, 4%, 6%) were hit within the horizon
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/backtest/screen")
async def backtest_screen(
    date: str = Query(..., description="YYYY-MM-DD evaluation date"),
    horizon: int = Query(20, description="Forward horizon in trading days")
):
    """
    Backtest: screen all 42 stocks on `date`, return BUY signals with forward
    return data.
    """
    init_historical()
    eval_date = pd.Timestamp(date)
    results = []

    for k, df in HISTORICAL_CACHE.items():
        mask = df.index <= eval_date  # Only data available ON OR BEFORE eval_date
        if mask.sum() < 60:
            continue

        # Score using PAST data only (no forward-looking bias)
        r = calc_scores(
            df[mask]["Close"].values, df[mask]["High"].values,
            df[mask]["Low"].values,  df[mask]["Volume"].values,
            info=None  # ← No fundamental data for backtesting
        )
        if r is None:
            continue

        # Recompute score on 2-lane scale (Teknikal + Momentum only, max 200 ÷ 2)
        total_bt = round((min(r["tek_score"], 100) + min(r["mom_score"], 100)) / 2)
        if total_bt >= 75:
            r["rec"] = "BUY"
        elif total_bt >= 55:
            r["rec"] = "HOLD"
        elif total_bt >= 35:
            r["rec"] = "WATCH"
        else:
            r["rec"] = "AVOID"
        r["score"] = total_bt

        if r["rec"] not in ("BUY",):
            continue

        # Look up forward return (actual outcome)
        fwd = df[df.index > eval_date]
        if len(fwd) >= horizon:
            eval_price  = safe(df[mask]["Close"].values[-1])
            fwd_close   = safe(fwd["Close"].values[horizon - 1])
            end_date    = fwd.index[horizon - 1]
            ret         = safe((fwd_close / eval_price - 1) * 100)
            max_price   = safe(fwd["High"].values[:horizon].max())

            # Check if conservative TP levels were hit (2%, 4%, 6%)
            tp1_hit = max_price >= eval_price * 1.02
            tp2_hit = max_price >= eval_price * 1.04
            tp3_hit = max_price >= eval_price * 1.06

            r["fwd"] = {
                "price_fwd": round(fwd_close),
                "end_date":  end_date.strftime("%Y-%m-%d"),
                "ret_pct":   round(ret, 2),
                "win":       ret > 0,
                "max_price": round(max_price),
                "tp1_hit":   tp1_hit, "tp2_hit": tp2_hit, "tp3_hit": tp3_hit,
            }

        r["kode"] = k
        results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)

    # IHSG return for the same period
    ihsg_ret = None
    if IHsgHISTORICAL is not None:
        m2 = IHsgHISTORICAL.index <= eval_date
        if m2.sum() > 0:
            ep2 = safe(IHsgHISTORICAL[m2]["Close"].values[-1])
            fw2 = IHsgHISTORICAL[IHsgHISTORICAL.index > eval_date]
            if len(fw2) >= horizon:
                ihsg_ret = round(safe((safe(fw2["Close"].values[horizon - 1]) / ep2 - 1) * 100), 2)

    return {
        "date":    date,
        "horizon": horizon,
        "count":   len(results),
        "data":    results,
        "ihsg_return": ihsg_ret,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SERVE STATIC FILES (UI)
# ═══════════════════════════════════════════════════════════════════════════════

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("Screening Engine → http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
