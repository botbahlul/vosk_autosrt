@echo off
setlocal EnableExtensions

REM ======================================================================
REM vosk_autosrt PyPI build script for Windows
REM
REM Supported:
REM   Windows 10 / 11
REM   Python 3.10+
REM
REM This script:
REM   1. Detects Python
REM   2. Checks Python version
REM   3. Updates setuptools and wheel
REM   4. Cleans previous build files
REM   5. Builds source distribution
REM   6. Builds Windows wheel
REM ======================================================================


echo.
echo ============================================================
echo vosk_autosrt PyPI BUILD
echo ============================================================
echo.


REM ----------------------------------------------------------------------
REM Find Python
REM ----------------------------------------------------------------------

set "PYTHON_CMD="

REM Prefer Python Launcher with Python 3.10
py -3.10 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.10"
    goto PYTHON_FOUND
)

REM Try Python 3
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto PYTHON_FOUND
)

echo ERROR: Python was not found.
echo.
echo Please install Python 3.10 or newer.
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
REM Check that setup.py exists
REM ----------------------------------------------------------------------

if not exist "setup.py" (
    echo.
    echo ERROR: setup.py was not found.
    echo.
    echo Please run this script from the vosk_autosrt project directory.
    echo.
    pause
    exit /b 1
)


REM ----------------------------------------------------------------------
REM Check package directory
REM ----------------------------------------------------------------------

if not exist "vosk_autosrt" (
    echo.
    echo ERROR: vosk_autosrt package directory was not found.
    echo.
    pause
    exit /b 1
)


REM ----------------------------------------------------------------------
REM Update build tools
REM ----------------------------------------------------------------------

echo.
echo ============================================================
echo Updating setuptools and wheel...
echo ============================================================
echo.

%PYTHON_CMD% -m pip install --upgrade setuptools wheel

if errorlevel 1 (
    echo.
    echo ERROR: Failed to update setuptools/wheel.
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

if exist "vosk_autosrt.egg-info" (
    echo Removing vosk_autosrt.egg-info...
    rmdir /s /q "vosk_autosrt.egg-info"
)


REM ----------------------------------------------------------------------
REM Build source distribution
REM ----------------------------------------------------------------------

echo.
echo ============================================================
echo Building source distribution...
echo ============================================================
echo.

%PYTHON_CMD% setup.py sdist

if errorlevel 1 (
    echo.
    echo ERROR: Source distribution build failed.
    echo.
    pause
    exit /b 1
)


REM ----------------------------------------------------------------------
REM Build Windows wheel
REM ----------------------------------------------------------------------

echo.
echo ============================================================
echo Building Windows wheel...
echo ============================================================
echo.

%PYTHON_CMD% setup.py bdist_wheel

if errorlevel 1 (
    echo.
    echo ERROR: Windows wheel build failed.
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
echo ============================================================
echo Finished.
echo ============================================================
echo.

endlocal
pause
