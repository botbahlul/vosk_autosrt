#!/bin/sh

# ======================================================================
# vosk_autosrt PyPI build script
#
# Supported:
#   Linux
#   macOS
#
# Linux:
#   Builds a normal wheel first, then uses auditwheel to create
#   a manylinux wheel suitable for PyPI.
#
# macOS:
#   Builds macOS x86_64 wheel for macOS 10.15.
# ======================================================================

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

echo
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
# Build tools
# ----------------------------------------------------------------------

echo
echo "Updating setuptools and wheel..."

"$PYTHON_CMD" -m pip install --upgrade setuptools wheel


# ----------------------------------------------------------------------
# Build source distribution
# ----------------------------------------------------------------------

echo
echo "Building source distribution..."

"$PYTHON_CMD" setup.py sdist


# ----------------------------------------------------------------------
# Build platform-specific wheel
# ----------------------------------------------------------------------

case "$OS_NAME" in

    Darwin)

        echo
        echo "Detected macOS."

        if [ "$CPU_ARCH" = "x86_64" ]; then

            echo "Detected Intel x86_64 macOS."
            echo "Building macOS 10.15 x86_64 wheel..."

            "$PYTHON_CMD" setup.py bdist_wheel \
                --plat-name macosx_10_15_x86_64

        else

            echo "Detected macOS architecture: $CPU_ARCH"
            echo "Building automatic macOS wheel..."

            "$PYTHON_CMD" setup.py bdist_wheel

        fi

        ;;


    Linux)

        echo
        echo "Detected Linux."

        if [ "$CPU_ARCH" != "x86_64" ]; then
            echo
            echo "ERROR: This build script currently targets Linux x86_64."
            echo "Detected architecture: $CPU_ARCH"
            exit 1
        fi


        # --------------------------------------------------------------
        # Build normal Linux wheel
        # --------------------------------------------------------------

        echo
        echo "Building Linux x86_64 wheel..."

        "$PYTHON_CMD" setup.py bdist_wheel


        # --------------------------------------------------------------
        # Install auditwheel
        # --------------------------------------------------------------

        echo
        echo "Checking auditwheel..."

        if ! command -v auditwheel >/dev/null 2>&1; then

            echo "auditwheel not found."
            echo "Installing auditwheel..."

            "$PYTHON_CMD" -m pip install --upgrade auditwheel

        fi


        # --------------------------------------------------------------
        # Repair Linux wheel
        # --------------------------------------------------------------

        echo
        echo "Running auditwheel..."

        mkdir -p dist/repaired


        "$PYTHON_CMD" -m auditwheel repair \
            --plat manylinux_2_17_x86_64 \
            --wheel-dir dist/repaired \
            dist/*linux_x86_64.whl


        # --------------------------------------------------------------
        # Replace original Linux wheel with repaired wheel
        # --------------------------------------------------------------

        echo
        echo "Replacing original Linux wheel..."

        rm -f dist/*linux_x86_64.whl

        mv dist/repaired/*.whl dist/

        rm -rf dist/repaired

        ;;


    *)

        echo
        echo "ERROR: Unsupported operating system: $OS_NAME"
        echo "This script supports Linux and macOS."
        exit 1

        ;;

esac


# ----------------------------------------------------------------------
# Check resulting distributions
# ----------------------------------------------------------------------

echo
echo "============================================================"
echo "Generated distributions"
echo "============================================================"
echo

ls -lh dist/


# ----------------------------------------------------------------------
# Verify wheel metadata
# ----------------------------------------------------------------------

echo
echo "Checking distributions with twine..."

if command -v twine >/dev/null 2>&1; then

    twine check dist/*

else

    echo "WARNING: twine is not installed."
    echo "Install it with:"
    echo
    echo "    $PYTHON_CMD -m pip install twine"
    echo

fi


echo
echo "============================================================"
echo "BUILD SUCCESSFUL"
echo "============================================================"
echo

