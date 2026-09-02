from __future__ import unicode_literals

import os
import platform
import sys
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


from vosk_autosrt import VERSION


# ----------------------------------------------------------------------
# Python version
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Platform detection
# ----------------------------------------------------------------------

SYSTEM = platform.system()
MACHINE = platform.machine().lower()


def get_platform_name():
    """
    Return a human-readable platform name.
    """
    if SYSTEM == "Windows":
        return "Windows"
    elif SYSTEM == "Linux":
        return "Linux"
    elif SYSTEM == "Darwin":
        return "macOS"
    else:
        return SYSTEM


def get_lib_files():
    """
    Return native Vosk library files required by this platform.
    """

    if SYSTEM == "Linux":
        return [
            "libvosk.so",
        ]

    elif SYSTEM == "Darwin":
        # Keep this as libvosk.dyld if that is the actual
        # filename shipped in vosk_autosrt.
        return [
            "libvosk.dyld",
        ]

    elif SYSTEM == "Windows":
        return [
            "libgcc_s_seh-1.dll",
            "libstdc++-6.dll",
            "libvosk.dll",
            "libwinpthread-1.dll",
        ]

    raise NotImplementedError(
        "Platform '{}' is not supported.".format(SYSTEM)
    )


# ----------------------------------------------------------------------
# Binary distribution
# ----------------------------------------------------------------------

class BinaryDistribution(Distribution):
    """
    Tell setuptools/wheel that this package contains
    platform-specific native libraries.
    """

    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


# ----------------------------------------------------------------------
# Verify native libraries
# ----------------------------------------------------------------------

def check_lib_files():
    """
    Check whether the native libraries expected for the current
    platform actually exist inside the vosk_autosrt package directory.
    """

    package_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vosk_autosrt",
    )

    missing = []

    for filename in get_lib_files():
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
        print("Machine  : {}".format(MACHINE))
        print()
        sys.exit(1)


check_lib_files()


# ----------------------------------------------------------------------
# Long description
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

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

    install_requires=[
        "sounddevice>=0.4.4",
        "vosk>=0.3.44",
        "requests>=2.3.0",
        "httpx>=0.13.3",
        "urllib3>=1.26.0,<3.0",
        "pysrt>=1.0.1",
        "six>=1.11.0",
        "av==12.2.0",
        "progressbar2>=3.34.3",
    ],

    license=open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "LICENSE",
        ),
        encoding="utf-8",
    ).read(),

    include_package_data=True,

    package_data={
        "vosk_autosrt": get_lib_files(),
    },

    distclass=BinaryDistribution,
)

