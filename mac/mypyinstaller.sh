pyinstaller --python=/usr/local/bin/python3.8 \
--add-data "./libvosk.dyld:." \
--hidden-import argparse \
--hidden-import sounddevice \
--hidden-import=sip --paths=/usr/local/lib/python3.8/site-packages/sipbuild \
--additional-hooks-dir=./ \
--onefile vosk_autosrt.py
