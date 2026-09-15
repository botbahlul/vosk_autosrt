#!/bin/sh

# ----------------------------------------------------------------------
# Find Python
# ----------------------------------------------------------------------
if [ -n "$PYTHON" ]; then
    PYTHON_CMD="$PYTHON"
#elif command -v python3.10 >/dev/null 2>&1; then
#    PYTHON_CMD="python3.10"
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
    sys.exit(1)
' || exit 1

# ----------------------------------------------------------------------
# Detect operating system
# ----------------------------------------------------------------------
OS_NAME="$(uname -s)"
CPU_ARCH="$(uname -m)"

# Normalize Windows uname output (Git Bash / MSYS2 / Cygwin) to "Windows"
case "$OS_NAME" in
    MINGW*|MSYS*|CYGWIN*) OS_NAME="Windows" ;;
esac

# Special detection for native Termux.
# /data/data/com.termux is also visible inside proot-distro,
# so its existence alone must NOT be used to identify Termux.
if [ "$OS_NAME" = "Linux" ]; then
    PYTHON_PREFIX="$("$PYTHON_CMD" -c 'import sys; print(sys.prefix)' 2>/dev/null)"

    case "$PYTHON_PREFIX" in
        /data/data/com.termux/files/usr*)
            OS_NAME="Android"
            ;;
    esac
fi

echo
echo "Operating system : $OS_NAME"
echo "Architecture     : $CPU_ARCH"

# ----------------------------------------------------------------------
# PyInstaller uses ':' as the src/dest separator on POSIX, ';' on Windows
# ----------------------------------------------------------------------
if [ "$OS_NAME" = "Windows" ]; then
    PYI_SEP=";"
else
    PYI_SEP=":"
fi

# ----------------------------------------------------------------------
# Clean previous build
# ----------------------------------------------------------------------
echo
echo "Cleaning previous build files..."

folder1="./build"
folder2="./dist"
file1="./vosk_autosrt.spec"

if [ -d "$folder1" ]; then
    rm -rf "$folder1"
fi

if [ -d "$folder2" ]; then
    rm -rf "$folder2"
fi

if [ -f "$file1" ]; then
    rm -f "$file1"
fi

# ----------------------------------------------------------------------
# Native library selection per platform / architecture
#
# POSIX sh has no arrays/dicts, so each branch below fills LIBVOSK_FILES
# with a space-separated list of filenames (none of these contain
# spaces, so word-splitting on it later is safe).
# ----------------------------------------------------------------------
case "$OS_NAME" in

    Windows)
        echo
        echo "Detected Windows ($CPU_ARCH)."
        LIBVOSK_FILES="libgcc_s_seh-1.dll libstdc++-6.dll libvosk.dll libwinpthread-1.dll"
        ;;

    Darwin)
        echo
        echo "Detected macOS."
        case "$CPU_ARCH" in
            x86_64|arm64)
                echo "Building macOS $CPU_ARCH executable binary..."
                LIBVOSK_FILES="libvosk.dyld"
                ;;
            *)
                echo "ERROR: Unsupported macOS architecture: $CPU_ARCH"
                exit 1
                ;;
        esac
        ;;

    Linux)
        echo
        echo "Detected Linux ($CPU_ARCH)."

        case "$CPU_ARCH" in
            x86_64|amd64)
                LIBVOSK_FILES="libvosk.so"
                ;;
            aarch64|arm64)
                LIBVOSK_FILES="libvosk_linux_aarch64.so"
                ;;
            armv7l|armv7|armhf)
                LIBVOSK_FILES="libvosk_linux_armv7l.so"
                ;;
            *)
                echo "ERROR: Unsupported Linux architecture: $CPU_ARCH"
                exit 1
                ;;
        esac
        ;;

    Android)
        echo
        echo "Detected Termux Android ($CPU_ARCH)."

        case "$CPU_ARCH" in
            aarch64|arm64)
                LIBVOSK_FILES="libvosk_android-arm64-v8a.so"
                ;;
            armv7l|armv7|armhf)
                LIBVOSK_FILES="libvosk_android-armeabi-v7a.so"
                ;;
            x86_64|amd64)
                LIBVOSK_FILES="libvosk_android-x86_64.so"
                ;;
            x86|i686|i386)
                LIBVOSK_FILES="libvosk_android-x86.so"
                ;;
            *)
                echo "ERROR: Unsupported Android architecture: $CPU_ARCH"
                exit 1
                ;;
        esac
        ;;

    *)
        echo
        echo "ERROR: Unsupported operating system: $OS_NAME"
        exit 1
        ;;
esac


# Make sure every listed native library actually exists before we build
for f in $LIBVOSK_FILES; do
    if [ ! -f "./$f" ]; then
        echo "ERROR: expected native library not found: ./$f"
        exit 1
    fi
done

# ----------------------------------------------------------------------
# Build the PyInstaller argument list
#
# We use "set --" to build the option list because POSIX sh has no
# arrays; this is the standard, quoting-safe way to accumulate
# variable-length argument lists in sh.
# ----------------------------------------------------------------------
set --

for f in $LIBVOSK_FILES; do
    set -- "$@" --add-data "./${f}${PYI_SEP}."
done

set -- "$@" \
    --hidden-import argparse \
    --hidden-import pysrt \
    --hidden-import six \
    --hidden-import progressbar \
    --hidden-import tqdm \
    --hidden-import requests \
    --hidden-import _cffi_backend \
    --hidden-import sounddevice


set -- "$@" \
    --additional-hooks-dir=./ \
    --onefile vosk_autosrt.py

echo
echo "Running: $PYTHON_CMD -m PyInstaller $*"
echo

"$PYTHON_CMD" -m PyInstaller "$@"
