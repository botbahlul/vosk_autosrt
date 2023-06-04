#!/usr/bin/env python
from __future__ import unicode_literals

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

long_description = (
    'voskautosrt is a utility for automatic speech recognition and subtitle generation. '
    'It takes a video or an audio file as input, performs an offline voice recognition '
    'generate transcriptions, and translates them to a different language and finally '
    'saves the resulting subtitles to disk. '
    'It supports 21 of input languages but can translate up to 134 lnguages  and can '
    'currently produce subtitles in SRT, VTT, JSON, and RAW format.'
)

setup(
    name="voskautosrt",
    version="0.0.1",
    description="a command line utility for automatic speech recognition and subtitle generation",
    long_description = long_description,
    author="Bot Bahlul",
    author_email="bot.bahlul@gmail.com",
    url="https://github.com/botbahlul/voskautosrt",
    packages=[str("voskautosrt")],
    entry_points={
        "console_scripts": [
            "voskautosrt = voskautosrt:main",
        ],
    },
    install_requires=[
        "requests>=2.3.0",
        "pysrt>=1.0.1",
        "progressbar2>=3.34.3",
        "six>=1.11.0",
        "httpx>=0.13.3",
        "ffmpeg_progress_yield>=0.7.2",
    ],
    license=open("LICENSE").read()
)
