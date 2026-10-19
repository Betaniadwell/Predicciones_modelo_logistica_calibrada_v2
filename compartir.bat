@echo off
REM ===================================================================
REM  Income Classifier Pro - iniciar todo y publicar con enlace publico
REM  Levanta: API (8000) + Web Streamlit (8501) + proxy (8080) + ngrok
REM ===================================================================
cd /d "%~dp0"

echo.
echo  Iniciando API (FastAPI)...
start "" cmd /c "uvicorn api:app --host 0.0.0.0 --port 8000"
timeout /t 4 /nobreak >nul

echo  Iniciando web (Streamlit)...
start "" cmd /c "streamlit run app.py"
timeout /t 6 /nobreak >nul

echo  Iniciando proxy inverso (8080)...
start "" cmd /c "node serve_publico.js"
timeout /t 3 /nobreak >nul

echo  Publicando con ngrok...
start "" cmd /c "ngrok http 8080"
timeout /t 8 /nobreak >nul

echo.
echo  ============================================================
echo   LOCAL      http://localhost:8501   (web)
echo              http://localhost:8000/docs (API)
echo.
echo   El enlace publico de ngrok aparece en:
echo     http://localhost:4040
echo   Si ves la pagina de aviso de ngrok, pulsa "Visit Site".
echo  ============================================================
echo.

start "" http://localhost:4040
