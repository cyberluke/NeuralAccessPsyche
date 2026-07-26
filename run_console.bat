@echo off
REM ---------------------------------------------------------------------------
REM Resilient Streamlit launcher for the NRAM Control Console.
REM
REM The console has been observed to die SILENTLY (no Python traceback, process
REM just exits back to the prompt). That signature indicates an EXTERNAL kill
REM (usually OS out-of-memory), not a code bug. SGLang's hierarchical KV cache
REM reserves ~15 GB of HOST RAM on top of the model, so long sessions under
REM memory pressure can get the Streamlit process reaped.
REM
REM This wrapper auto-restarts the server whenever it exits, and tees all output
REM to streamlit_server.log so the last messages before a death are preserved.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set LOG=streamlit_server.log
set PORT=8501

echo [nram-console] Starting resilient Streamlit launcher. Logs: %LOG%
echo [nram-console] Press Ctrl+C to stop.

:loop
echo. >> "%LOG%"
echo ===== [nram-console] (re)starting streamlit at %date% %time% ===== >> "%LOG%"
echo [nram-console] (re)starting streamlit at %time%
call streamlit run streamlit_console.py --server.port %PORT% --server.headless true >> "%LOG%" 2>&1

echo [nram-console] streamlit exited with code %ERRORLEVEL% at %time% >> "%LOG%"
echo [nram-console] streamlit exited with code %ERRORLEVEL%. Restarting in 3s... (Ctrl+C to abort)
timeout /t 3 /nobreak >nul
goto loop
