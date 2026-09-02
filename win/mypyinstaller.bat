@echo off
setlocal EnableExtensions

REM ======================================================================
REM vosk_autosrt PyInstaller build script for Windows
REM
REM Requirements:
REM   - Python 3.10+
REM   - PyInstaller 6.x
REM   - setuptools 82+ is supported
REM
REM This script:
REM   1. Detects Python 3.10
REM   2. Checks Python version
REM   3. Updates PyInstaller and hooks-contrib
REM   4. Cleans previous build files
REM   5. Builds a one-file Windows executable
REM ======================================================================


echo.
echo ============================================================
echo vosk_autosrt PyInstaller BUILD
echo ============================================================
echo.


REM ----------------------------------------------------------------------
REM Find Python
REM ----------------------------------------------------------------------

set "PYTHON_CMD="

REM Prefer Python 3.10
py -3.10 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.10"
    goto PYTHON_FOUND
)

REM Fallback to python
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto PYTHON_FOUND
)

echo ERROR: Python was not found.
echo.
echo Python 3.10 or newer is required.
echo.
pause
exit /b 1


:PYTHON_FOUND

echo Using Python:
%PYTHON_CMD% --version
echo.


REM ----------------------------------------------------------------------
REM Check Python version
REM ----------------------------------------------------------------------

%PYTHON_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"

if errorlevel 1 (
    echo.
    echo ERROR: Python 3.10 or newer is required.
    echo.
    %PYTHON_CMD% --version
    echo.
    pause
    exit /b 1
)


REM ----------------------------------------------------------------------
REM Check source file
REM ----------------------------------------------------------------------

if not exist "vosk_autosrt.py" (
    echo.
    echo ERROR: vosk_autosrt.py was not found.
    echo.
    echo Please run this script from the vosk_autosrt project directory.
    echo.
    pause
    exit /b 1
)


REM ----------------------------------------------------------------------
REM Update PyInstaller
REM ----------------------------------------------------------------------

echo.
echo ============================================================
echo Updating PyInstaller...
echo ============================================================
echo.

%PYTHON_CMD% -m pip install --upgrade pyinstaller pyinstaller-hooks-contrib

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install/update PyInstaller.
    echo.
    pause
    exit /b 1
)


REM ----------------------------------------------------------------------
REM Display PyInstaller version
REM ----------------------------------------------------------------------

echo.
echo PyInstaller version:
%PYTHON_CMD% -m PyInstaller --version

if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller is not working correctly.
    echo.
    pause
    exit /b 1
)


REM ----------------------------------------------------------------------
REM Clean previous build
REM ----------------------------------------------------------------------

echo.
echo ============================================================
echo Cleaning previous build files...
echo ============================================================
echo.

if exist "build" (
    echo Removing build...
    rmdir /s /q "build"
)

if exist "dist" (
    echo Removing dist...
    rmdir /s /q "dist"
)

REM Remove all generated .spec files
del /q "*.spec" >nul 2>&1


REM ----------------------------------------------------------------------
REM Check required Vosk DLL files
REM ----------------------------------------------------------------------

echo.
echo ============================================================
echo Checking Vosk DLL files...
echo ============================================================
echo.

set "DLL_ERROR=0"

if not exist "libgcc_s_seh-1.dll" (
    echo ERROR: libgcc_s_seh-1.dll not found.
    set "DLL_ERROR=1"
)

if not exist "libstdc++-6.dll" (
    echo ERROR: libstdc++-6.dll not found.
    set "DLL_ERROR=1"
)

if not exist "libvosk.dll" (
    echo ERROR: libvosk.dll not found.
    set "DLL_ERROR=1"
)

if not exist "libwinpthread-1.dll" (
    echo ERROR: libwinpthread-1.dll not found.
    set "DLL_ERROR=1"
)

if "%DLL_ERROR%"=="1" (
    echo.
    echo ERROR: One or more required Vosk DLL files are missing.
    echo.
    pause
    exit /b 1
)

echo All required Vosk DLL files found.


REM ----------------------------------------------------------------------
REM Build executable
REM ----------------------------------------------------------------------

echo.
echo ============================================================
echo Building vosk_autosrt.exe ...
echo ============================================================
echo.


%PYTHON_CMD% -m PyInstaller ^
    --clean ^
    --onefile ^
    --name vosk_autosrt ^
    --add-data "libgcc_s_seh-1.dll;." ^
    --add-data "libstdc++-6.dll;." ^
    --add-data "libvosk.dll;." ^
    --add-data "libwinpthread-1.dll;." ^
    --hidden-import argparse ^
    --hidden-import pysrt ^
    --hidden-import six ^
    --hidden-import progressbar ^
    --hidden-import tqdm ^
    --hidden-import requests ^
    --hidden-import _cffi_backend ^
    --hidden-import sounddevice ^
    --additional-hooks-dir=. ^
    vosk_autosrt.py


if errorlevel 1 (
    echo.
    echo ============================================================
    echo BUILD FAILED
    echo ============================================================
    echo.
    pause
    exit /b 1
)


REM ----------------------------------------------------------------------
REM Result
REM ----------------------------------------------------------------------

echo.
echo ============================================================
echo BUILD SUCCESSFUL
echo ============================================================
echo.

echo Generated files:
echo.

dir /b "dist"

echo.
echo Executable:
echo.
echo     dist\vosk_autosrt.exe
echo.

echo ============================================================
echo Finished.
echo ============================================================
echo.

endlocal
pause
