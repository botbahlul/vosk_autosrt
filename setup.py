from __future__ import unicode_literals

import os
import platform
import re
import sys
import sysconfig
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="setuptools",
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="setuptools",
)

warnings.filterwarnings(
    "ignore",
    message=".*is deprecated.*",
)


try:
    from setuptools import setup, find_packages
    from setuptools.dist import Distribution
except ImportError:
    print("ERROR: setuptools is required to build vosk_autosrt.")
    print("Please install it with:")
    print("    python -m pip install setuptools wheel")
    sys.exit(1)


# ======================================================================
# Python version
# ======================================================================

MIN_PYTHON = (3, 10)

if sys.version_info < MIN_PYTHON:
    print(
        "THIS MODULE REQUIRES PYTHON 3.10+."
    )
    print(
        "YOU ARE CURRENTLY USING PYTHON {0}".format(
            platform.python_version()
        )
    )
    sys.exit(1)


# ======================================================================
# Get package version WITHOUT importing vosk_autosrt
# ======================================================================

PACKAGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "vosk_autosrt",
)

INIT_FILE = os.path.join(
    PACKAGE_DIR,
    "__init__.py",
)

with open(INIT_FILE, encoding="utf-8") as f:
    INIT_CONTENT = f.read()

VERSION_MATCH = re.search(
    r'^VERSION\s*=\s*[\'"]([^\'"]+)[\'"]',
    INIT_CONTENT,
    re.MULTILINE,
)

if not VERSION_MATCH:
    print(
        "ERROR: Unable to find VERSION in "
        "vosk_autosrt/__init__.py"
    )
    sys.exit(1)

VERSION = VERSION_MATCH.group(1)


# ======================================================================
# Platform detection
# ======================================================================

SYSTEM = platform.system()
MACHINE = platform.machine().lower()


def _is_native_termux():
    """
    Detect ONLY native Termux.

    IMPORTANT:
    /data/data/com.termux is visible from proot-distro, so it must NOT
    be used by itself to identify native Termux.

    Native Termux Python normally has a prefix/executable under:
        /data/data/com.termux/files/usr

    A Python running inside proot-distro has a Linux prefix such as:
        /root/venv
        /usr
        /usr/local
    """

    termux_prefix = "/data/data/com.termux/files/usr"

    # Python installation prefix
    for value in (
        getattr(sys, "prefix", ""),
        getattr(sys, "base_prefix", ""),
        getattr(sys, "exec_prefix", ""),
        getattr(sys, "base_exec_prefix", ""),
    ):
        if value.startswith(termux_prefix):
            return True

    # Python executable
    executable = os.path.realpath(sys.executable)
    if executable.startswith(termux_prefix + "/"):
        return True

    # sysconfig prefix/platform
    try:
        config_prefix = sysconfig.get_config_var("prefix") or ""
        if config_prefix.startswith(termux_prefix):
            return True
    except Exception:
        pass

    return False


IS_TERMUX = _is_native_termux()


def get_platform_name():
    if IS_TERMUX:
        return "Android (Termux)"
    elif SYSTEM == "Windows":
        return "Windows"
    elif SYSTEM == "Linux":
        return "Linux"
    elif SYSTEM == "Darwin":
        return "macOS"
    else:
        return SYSTEM


# ======================================================================
# Native library groups
# ======================================================================

WINDOWS_LIBS = [
    "libgcc_s_seh-1.dll",
    "libstdc++-6.dll",
    "libvosk.dll",
    "libwinpthread-1.dll",
]


LINUX_ARCH_LIB_MAP = {
    "x86_64": "libvosk.so",
    "amd64": "libvosk.so",

    "aarch64": "libvosk_linux_aarch64.so",
    "arm64": "libvosk_linux_aarch64.so",

    "armv7l": "libvosk_linux_armv7l.so",
    "armv7": "libvosk_linux_armv7l.so",
    "armhf": "libvosk_linux_armv7l.so",
}


ANDROID_ARCH_LIB_MAP = {
    "aarch64": "libvosk_android-arm64-v8a.so",
    "arm64": "libvosk_android-arm64-v8a.so",

    "armv7l": "libvosk_android-armeabi-v7a.so",
    "armv7": "libvosk_android-armeabi-v7a.so",
    "armhf": "libvosk_android-armeabi-v7a.so",

    "x86_64": "libvosk_android-x86_64.so",
    "amd64": "libvosk_android-x86_64.so",

    "x86": "libvosk_android-x86.so",
    "i686": "libvosk_android-x86.so",
    "i386": "libvosk_android-x86.so",
}


DARWIN_LIBS = [
    "libvosk.dyld",
]


# sdist must contain every native binary.
LINUX_LIBS_ALL = list(dict.fromkeys(LINUX_ARCH_LIB_MAP.values()))
ANDROID_LIBS_ALL = list(dict.fromkeys(ANDROID_ARCH_LIB_MAP.values()))

ALL_LIBS = (
    WINDOWS_LIBS
    + LINUX_LIBS_ALL
    + ANDROID_LIBS_ALL
    + DARWIN_LIBS
)


# ======================================================================
# Select native libraries
# ======================================================================

def get_lib_files():
    """
    Return native libraries for the current build.

    sdist:
        Include ALL platform/architecture libraries.

    wheel/local install:
        Include only the native library matching the current
        operating system and architecture.
    """

    is_sdist = "sdist" in sys.argv

    if is_sdist:
        return ALL_LIBS

    # --------------------------------------------------------------
    # Native Termux / Android
    # --------------------------------------------------------------
    if IS_TERMUX:
        lib = ANDROID_ARCH_LIB_MAP.get(MACHINE)

        if lib is None:
            raise NotImplementedError(
                "Unsupported Android/Termux architecture: {}".format(
                    MACHINE
                )
            )

        return [lib]

    # --------------------------------------------------------------
    # Windows
    # --------------------------------------------------------------
    if SYSTEM == "Windows":
        return WINDOWS_LIBS

    # --------------------------------------------------------------
    # macOS
    # --------------------------------------------------------------
    if SYSTEM == "Darwin":
        return DARWIN_LIBS

    # --------------------------------------------------------------
    # Linux / glibc
    # --------------------------------------------------------------
    if SYSTEM == "Linux":
        lib = LINUX_ARCH_LIB_MAP.get(MACHINE)

        if lib is None:
            raise NotImplementedError(
                "Unsupported Linux architecture: {}".format(
                    MACHINE
                )
            )

        return [lib]

    raise NotImplementedError(
        "Platform '{}' is not supported.".format(SYSTEM)
    )


# ======================================================================
# Binary distribution
# ======================================================================

class BinaryDistribution(Distribution):
    """
    Tell setuptools/wheel that this package contains
    platform-specific native libraries.
    """

    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


# ======================================================================
# Verify native libraries
# ======================================================================

def check_lib_files():
    """
    Check whether the native libraries expected for the current
    command exist.
    """

    package_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vosk_autosrt",
    )

    expected = get_lib_files()

    missing = []

    for filename in expected:
        filepath = os.path.join(package_dir, filename)

        if not os.path.isfile(filepath):
            missing.append(filepath)

    if missing:
        print()
        print("ERROR: Required native library file(s) not found:")

        for filepath in missing:
            print("  - {}".format(filepath))

        print()
        print("Platform : {}".format(get_platform_name()))
        print("System   : {}".format(SYSTEM))
        print("Machine  : {}".format(MACHINE))
        print("Python   : {}".format(sys.executable))
        print("Prefix   : {}".format(sys.prefix))
        print()

        sys.exit(1)


check_lib_files()


# ======================================================================
# Long description
# ======================================================================

long_description = (
    "vosk_autosrt is a COMMAND LINE UTILITY for automatic speech "
    "recognition and subtitle generation using Vosk API. It takes "
    "video or audio files as input, converts them to temporary wav "
    "files, then performs offline voice recognition, generates "
    "transcriptions, and optionally translates them to different "
    "languages and finally saves the resulting subtitles to disk. "
    "It supports 21 input languages but can translate up to 134 "
    "languages and can produce subtitles currently in SRT, VTT, "
    "JSON, and RAW format."
)


# ======================================================================
# Runtime dependencies
# ======================================================================

INSTALL_REQUIRES = [
    "audioop-lts; python_version >= '3.13'",
    "sounddevice>=0.4.4",
    "requests>=2.3.0",
    "httpx>=0.13.3",
    "urllib3>=1.26.0,<3.0",
    "pysrt>=1.0.1",
    "six>=1.11.0",
    "progressbar2>=3.34.3",
]


# ======================================================================
# Setup
# ======================================================================

setup(
    name="vosk_autosrt",

    version=VERSION,

    description=(
        "a command line utility for automatic speech recognition "
        "and subtitle generation"
    ),

    long_description=long_description,

    author="Bot Bahlul",
    author_email="bot.bahlul@gmail.com",

    url="https://github.com/botbahlul/vosk_autosrt",

    packages=find_packages(),

    entry_points={
        "console_scripts": [
            "vosk_autosrt=vosk_autosrt:main",
        ],
    },

    install_requires=INSTALL_REQUIRES,

    license=open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "LICENSE",
        ),
        encoding="utf-8",
    ).read(),

    include_package_data=False,

    package_data={
        "vosk_autosrt": get_lib_files(),
    },

    distclass=BinaryDistribution,
)