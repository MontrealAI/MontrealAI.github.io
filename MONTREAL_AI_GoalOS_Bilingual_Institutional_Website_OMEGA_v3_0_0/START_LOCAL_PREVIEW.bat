@echo off
cd /d "%~dp0"
py -3 start_local_preview.py || python start_local_preview.py
