# 🎯 Screening Engine — Rule-Based IDX Stock Screener

> **"Satu terminal. Screening saham IDX, tanpa ML, dengan backtest."**

Platform all-in-one untuk analisis saham Indonesia: **Dashboard IHSG**, **Analisis 3 Jalur** (Teknikal + Momentum + Fundamental), **Trading Plan ATR**, **Rekomendasi Batch**, dan **Backtest Time Machine**.

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-production-brightgreen)

---

## ✨ Fitur

### 🏠 Dashboard IHSG
- Harga IHSG real-time dengan status pasar (Hijau/Kuning/Merah — faktual, bukan prediksi)
- Grafik candlestick 1 tahun + MA20 & MA50 overlay
- Top 10 Gainers & Losers dengan sektor masing-masing
- **Market Overview**: breadth, foreign flow proxy, sentiment, technical outlook

### 🔍 Analisis Per Emiten
- **Score komposit 0–100** dari 3 jalur:
  - 🔵 **Teknikal** (9 indikator: MA, RSI, MACD, Bollinger, ATR, Stoch, ADX, EMA, Volume)
  - 🟠 **Momentum** (5 sinyal: NR7, BB Squeeze, Volume Spike, Return, Near High)
  - 🟢 **Fundamental** (7 rasio: PER, PBV, ROE, Profit Margin, Revenue Growth, DER, Div Yield)
- Candlestick chart interaktif dengan timeframe 1B/3B/6B/1T + volume + MA overlay
- Detail setiap indikator (sinyal, label, skor)

### 🎯 Trading Plan (ATR-Based Conservative)
- **Range Entry Rekomendasi** — harga ideal untuk masuk posisi
- **Stop Loss** — 1.5× ATR di bawah harga
- **TP1, TP2, TP3** — target take profit 1× / 1.8× / 2.5× ATR
- **Estimasi hari** ke tiap level TP + R/R ratio

### 📊 Rekomendasi Batch
- Screening 12+ saham sekaligus dengan ranking skor
- Mini bar 3 warna untuk perbandingan cepat antar jalur
- Grade A/B/C/D + "grade" label

### ⏰ Backtest Time Machine
- Pilih tanggal di 2025 → lihat sinyal BUY & proyeksi TP
- Klik "Cek Hasil Aktual" → lihat return riil & TP mana yang kena
- Sama persis dengan scoring engine real-time (teknikal + momentum, tanpa fundamental)
- 42 emiten, 118 tanggal backtest, range 5–30 hari

---

## 🚀 Cara Pakai

### Prasyarat
- Python 3.11+
- Git (opsional)

### 1. Clone repo
```bash
git clone https://github.com/Attaboy473/screening-engine.git
cd screening-engine
```

### 2. Install dependencies
```bash
pip install yfinance ta fastapi uvicorn pandas numpy
```

### 3. Jalankan
```bash
python main.py
```

### 4. Buka browser
```
http://localhost:8000
```

> 💡 **Pro tip:** Double-klik `RUN_SCREENER.bat` di folder project untuk auto-start + auto-buka browser.

---

## 🧠 Cara Kerja Scoring

```
Input: Kode saham "BBCA"
        │
        ▼
┌────────────────────────────────┐
│ 1. Fetch OHLCV + Fundamental   │  ← Yahoo Finance (yfinance)
└──────────┬─────────────────────┘
           ▼
┌────────────────────────────────┐
│ 2. Hitung 21 indikator         │
│    Teknikal + Momentum + Fund. │
└──────────┬─────────────────────┘
           ▼
┌────────────────────────────────┐
│ 3. Composite Scoring (0–100)   │
│    (Tek + Mom + Fun) / 3       │
└──────────┬─────────────────────┘
           ▼
┌────────────────────────────────┐
│ 4. Likuiditas Filter           │
│    Grade A/B/C/D               │
└──────────┬─────────────────────┘
           ▼
┌────────────────────────────────┐
│ 5. Rekomendasi + Trading Plan  │
│    Entry Range, SL, TP1-3      │
└────────────────────────────────┘
```

---

## 📐 Threshold Rekomendasi (0–100)

| Score | Kategori | Rekomendasi |
|------:|----------|-------------|
| 80–100 | Sangat Baik | 🟢 **STRONG BUY** |
| 63–79 | Baik | 🔵 **BUY** |
| 47–62 | Netral | 🟡 **HOLD** |
| 30–46 | Buruk | 🟠 **WATCH** |
| 0–29 | Jelek | 🔴 **AVOID** |

---

## 📊 Hasil Backtest 2025

| Metrik | Hasil |
|--------|-------|
| **TP1 Hit Rate** | **78.9%** |
| **TP2 Hit Rate** | **56.3%** |
| **Win Rate (BUY)** | **54.9%** |
| **Observasi** | 2,184 data point |

> TP1 (target paling konservatif, 1× ATR) tercapai hampir **4 dari 5 kali**.

---

## 🗂 Struktur File

```
screening-engine/
├── main.py              # Backend FastAPI + Scoring Engine + Backtest
├── static/
│   └── index.html        # Single-page UI (Dashboard + Analisis + Rekomendasi + Backtest)
├── README.md             # Dokumentasi ini
├── .gitignore
└── RUN_SCREENER.bat      # One-click launcher
```

---

## 🛠 Tech Stack

| Lapisan | Teknologi |
|---------|-----------|
| **Backend** | Python 3.11 + FastAPI + Uvicorn |
| **Data** | Yahoo Finance (`yfinance`) |
| **Indikator** | `ta` (Technical Analysis Library) |
| **Chart** | TradingView Lightweight Charts v4.1.3 (CDN) |
| **UI** | Vanilla HTML/CSS/JS (no framework) |

---

## 🔑 Key Design Decisions

1. **Rule-based, not ML** — transparan, bisa dijelaskan ke stakeholder
2. **Skala 0–100** — mudah dipahami, bukan skala 300 yang membingungkan
3. **ATR-based Conservative TP** — TP dekat & realistis (78.9% hit rate)
4. **No look-ahead in backtest** — fundamental excluded, hanya data yang tersedia saat itu
5. **Fixed-width recommendation cards** — kolom sejajar vertikal, rapi presentasi

---

## ⚠️ Disclaimer

Aplikasi ini adalah **alat bantu screening** berbasis aturan teknikal. **BUKAN** rekomendasi investasi. Data berasal dari Yahoo Finance (delay ~15 menit untuk saham IDX). Selalu lakukan riset mandiri (DYOR) dan kelola risiko sebelum berinvestasi.

---

## 📝 Lisensi

MIT — bebas pakai, modifikasi, dan distribusi.

---

Dibuat dengan ❤️ oleh [Attaboy473](https://github.com/Attaboy473)
