@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
uvicorn ia_service:app --host 127.0.0.1 --port 8000