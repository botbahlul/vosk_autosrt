#!/bin/sh

# ----------------------------------------------------------------------
# vosk_autosrt PyPI build script
#
# Supported:
#   Linux
#   macOS
#
# Windows:
#   Use setup.py directly or run the equivalent commands from
#   PowerShell / CMD.
# ----------------------------------------------------------------------

set -e

# ----------------------------------------------------------------------
# Find Python
# ----------------------------------------------------------------------

if [ -n "$PYTHON" ]; then
    PYTHON_CMD="$PYTHON"
elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON_CMD="python3.10"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python 3 was not found."
    exit 1
fi


# ----------------------------------------------------------------------
# Check Python version
# ----------------------------------------------------------------------

echo "Using Python:"
"$PYTHON_CMD" --version

"$PYTHON_CMD" -c '
import sys

if sys.version_info < (3, 10):
    print("ERROR: Python 3.10 or newer is required.")
    print(
        "Current version: {}.{}.{}".format(
            sys.version_info[0],
            sys.version_info[1],
            sys.version_info[2],
        )
    )
    sys.exit(1)
'


# ----------------------------------------------------------------------
# Check required build tools
# ----------------------------------------------------------------------

echo
echo "Checking build tools..."

"$PYTHON_CMD" -m pip install --upgrade setuptools wheel


# ----------------------------------------------------------------------
# Detect operating system
# ----------------------------------------------------------------------

OS_NAME="$(uname -s)"
CPU_ARCH="$(uname -m)"

echo
echo "Operating system : $OS_NAME"
echo "Architecture     : $CPU_ARCH"


# ----------------------------------------------------------------------
# Clean previous build
# ----------------------------------------------------------------------

echo
echo "Cleaning previous build files..."

rm -rf build
rm -rf dist
rm -rf vosk_autosrt.egg-info


# ----------------------------------------------------------------------
# Build source distribution
# ----------------------------------------------------------------------

echo
echo "Building source distribution..."

"$PYTHON_CMD" setup.py sdist


# ----------------------------------------------------------------------
# Build platform-specific wheel
# ----------------------------------------------------------------------

echo
echo "Building wheel..."

case "$OS_NAME" in

    Darwin)

        # macOS
        #
        # If this machine is Intel x86_64 and the package is intended
        # to run on macOS Catalina 10.15, explicitly use:
        #
        #   macosx_10_15_x86_64
        #
        # Otherwise let wheel determine the correct platform tag.

        if [ "$CPU_ARCH" = "x86_64" ]; then

            echo "Detected Intel macOS (x86_64)."
            echo "Building wheel for macOS 10.15 x86_64..."

            "$PYTHON_CMD" setup.py bdist_wheel \
                --plat-name macosx_10_15_x86_64

        else

            echo "Detected macOS architecture: $CPU_ARCH"
            echo "Using automatic wheel platform tag..."

            "$PYTHON_CMD" setup.py bdist_wheel

        fi
        ;;

    Linux)

        echo "Detected Linux."
        echo "Using automatic Linux wheel platform tag..."

        "$PYTHON_CMD" setup.py bdist_wheel
        ;;

    *)

        echo
        echo "ERROR: Unsupported shell platform: $OS_NAME"
        echo "This script supports Linux and macOS."
        exit 1
        ;;

esac


# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------

echo
echo "============================================================"
echo "BUILD SUCCESSFUL"
echo "============================================================"
echo
echo "Generated files:"
ls -lh dist/
echo

