from __future__ import unicode_literals
import sys
import platform
import os
import stat
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='setuptools')

try:
    from setuptools import setup, find_packages
    from setuptools.dist import Distribution
except ImportError:
    from distutils.core import setup
    from distutils.dist import Distribut

from vosk_autosrt import VERSION

if sys.version_info <= (3, 8):
    print("THIS MODULE REQUIRES PYTHON 3.8+. YOU ARE CURRENTLY USING PYTHON {0}".format(sys.version))
    sys.exit(1)

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False

    def get_ext_modules(self):
        if platform.system() == 'Windows':
            return []
        else:
            return super().get_ext_modules()

def get_lib_files():
    if platform.system() == 'Linux':
        return ['libvosk.so']
    elif platform.system() == 'Darwin':
        return ['libvosk.dyld']
    elif platform.system() == 'Windows':
        return ['libgcc_s_seh-1.dll', 'libstdc++-6.dll', 'libvosk.dll', 'libwinpthread-1.dll']
    if not (platform.system() == 'Linux' or platform.system() == 'Darwin' or platform.system() == 'Windows'):
        raise NotImplementedError(f"Platform '{platform.system()}' is not supported.")

long_description = (
    'vosk_autosrt is a COMMAND LINE UTILLITY for automatic speech recognition and subtitle generation using  '
    'Vosk API. It takes video or audio files as input, convert them to temporary wav files then performs an  '
    'offline voice recognition, generate transcriptions, and optionally translates them to different language'
    'and finally save the resulting subtitles to disk.'
    'It supports 21 input languages but can translate up to 134 languages and can produce subtitles currently'
    'in SRT, VTT, JSON, and RAW format.'
)

setup(
    name="vosk_autosrt",
    version=VERSION,
    description="a command line utility for automatic speech recognition and subtitle generation",
    long_description = long_description,
    author="Bot Bahlul",
    author_email="bot.bahlul@gmail.com",
    url="https://github.com/botbahlul/vosk_autosrt",
    packages=[str("vosk_autosrt")],
    entry_points={
        "console_scripts": [
            "vosk_autosrt = vosk_autosrt:main",
        ],
    },
    install_requires=[
        "sounddevice>=0.4.4",
        "vosk>=0.3.44",
        "requests>=2.3.0",
        "httpx>=0.13.3",
        "urllib3 >=1.26.0,<3.0",
        "pysrt>=1.0.1",
        "six>=1.11.0",
        "progressbar2>=3.34.3",
    ],
    license=open("LICENSE").read(),
    include_package_data=True,
    package_data={'': get_lib_files()},
    distclass=BinaryDistribution,
)
