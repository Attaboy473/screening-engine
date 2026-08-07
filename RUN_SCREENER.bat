@echo off
cd /d "C:\Users\USER\Desktop\Hermes\Random\Screening Engine"
echo.
echo ============================================
echo   Screening Engine — All-in-One (port 8000)
echo ============================================
echo.
echo   Dashboard · Analisis · Rekomendasi · Backtest
echo   Browser akan terbuka otomatis.
echo.
echo   Tekan Ctrl+C untuk berhenti.
echo ============================================
echo.
start "" http://localhost:8000
python main.py
pause
