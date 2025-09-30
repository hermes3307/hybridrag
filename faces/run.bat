@echo off
REM Windows Batch Launcher for Face Recognition System
REM Usage: run.bat [1-10] or just run.bat for menu

setlocal enabledelayedexpansion

if "%~1"=="" (
    REM No argument - show launcher menu
    python 0_launcher.py
    goto :end
)

REM Map number to script
set "script="
if "%~1"=="0" set "script=0_launcher.py"
if "%~1"=="1" set "script=1_setup_database.py"
if "%~1"=="2" set "script=2_database_info.py"
if "%~1"=="3" set "script=3_download_faces.py"
if "%~1"=="4" set "script=4_download_faces_gui.py"
if "%~1"=="5" set "script=5_embed_faces.py"
if "%~1"=="6" set "script=6_embed_faces_gui.py"
if "%~1"=="7" set "script=7_search_faces_gui.py"
if "%~1"=="8" set "script=8_validate_embeddings.py"
if "%~1"=="9" set "script=9_test_features.py"
if "%~1"=="10" set "script=10_complete_demo.py"

if "!script!"=="" (
    echo Invalid component number: %~1
    echo Usage: run.bat [0-10]
    echo   0 - Launcher menu
    echo   1 - Setup database
    echo   2 - Database info
    echo   3 - Download faces ^(CLI^)
    echo   4 - Download faces ^(GUI^)
    echo   5 - Embed faces ^(CLI^)
    echo   6 - Embed faces ^(GUI^)
    echo   7 - Search faces ^(GUI^)
    echo   8 - Validate embeddings
    echo   9 - Test features
    echo   10 - Complete demo
    goto :end
)

REM Run the script
python !script! %2 %3 %4 %5 %6 %7 %8 %9

:end
endlocal