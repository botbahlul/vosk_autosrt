#!/bin/sh

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

/usr/local/bin/python3.10 -m PyInstaller \
    --add-data "./libvosk.dyld:." \
    --hidden-import argparse \
    --hidden-import pysrt \
    --hidden-import six \
    --hidden-import progressbar \
    --hidden-import tqdm \
    --hidden-import requests \
    --hidden-import _cffi_backend \
    --hidden-import sounddevice \
    --hidden-import=sip --paths=/usr/local/lib/python3.10/site-packages/sipbuild \
    --additional-hooks-dir=./ \
    --onefile vosk_autosrt.py
