# vosk_autosrt <a href="https://pypi.python.org/pypi/vosk_autosrt"><img src="https://img.shields.io/pypi/v/vosk_autosrt.svg"></img></a>
  
### Auto-generated subtitles for any video
vosk_autosrt is a simple command line tool made with python to auto generate subtitle/closed caption file for any video or audio file using VOSK API and translate it automaticly for free using unofficial Google Translate API.

### Installation
If you don't have python on your Windows system you can try the compiled version from this git release assets
https://github.com/botbahlul/vosk_autosrt/releases

Just extract those ffmpeg.exe, ffprobe.exe, and vosk_autosrt.exe into a folder that has been added to PATH ENVIRONTMET for example in C:\Windows\system32

You can get latest version of ffmpeg and ffprobe from https://www.ffmpeg.org/

In Linux you have to install this script with python (version minimal 3.8) and install ffmpeg with your linux package manager for example in debian based linux distribution you can type :

```
apt update
apt install -y ffmpeg
```

To install this vosk_autosrt, just type :
```
pip install vosk_autosrt
```

When you run this app for the very first time it may takes some times to download vosk language model, you can check those  downloaded models in "/home/username/.cache/vosk" (if you're on Linux) and "C:\\Users\\Username\\.cache\vosk\\" (if you're on Windows).

You can always download those small models manually from https://alphacephei.com/vosk/models then extract them to that used folder.
Pay attension to those folders name, because their names should contain the language codes in \"\_\_init\_\_.py\", esspecially for Chinese language, which in \"\_\_init\_\_.py\" its code is \"zh\", so you should rename that extracted downloaded model to \"vosk-model-small-zh-0.22\". This is needed for GoogleTranslate funcion to work properly.
![image](https://user-images.githubusercontent.com/88623122/234000963-c2ab4c69-70fd-4374-9a1a-0cc1316791e8.png)
![image](https://user-images.githubusercontent.com/88623122/234001411-f06821c9-0b68-4414-b3c3-a7280da4d560.png)

You can try to compile that vosk_autosrt.py script in win/linux/mac folder into a single executable file with that mypyinstaller.bat/mypyinstaller.sh:
```
pip install pyinstaller
mypyinstaller.bat
```

The executable compiled file will be placed by pyinstaller into dist subfolder of your current working folder, so you can just rename and put that compiled file into a folder that has been added to your PATH ENVIRONTMENT so you can execute it from anywhere

I was succesfuly compiled it in Windows 10 with pyinstaller-5.1 and Pyhton-3.10.4, and python-3.8.12 in Debian 9

Another alternative way to install this script with python is by cloning this git (or downloading this git as zip then extract it into a folder), and then just type :

```
pip install wheel
python setup.py bdist_wheel
```

Then check the name of the whl file created in dist folder. In case the filename is vosk_autosrt-0.0.2-cp310-cp310-win_amd64.whl then you can install that whl file with pip :
```
cd dist
pip install vosk_autosrt-0.0.2-cp310-cp310-win_amd64.whl
```

You can also install this script (or any pip package) in ANDROID DEVICES via PYTHON package in TERMUX APP

https://github.com/termux/termux-app/releases/tag/v0.118.0

Choose the right apk for your device, install it, then open it

Type these commands to get python, pip, this vosk_autosrt, (and any other pip packages) :

```
termux-setup-storage
pkg update -y
pkg install -y python
pkg install -y ffmpeg
pip install vosk_autosrt
```

### Simple usage example 

```
vosk_autosrt --list-src-languages
vosk_autosrt -S zh -D en "Episode 1.mp4"
```

If you don't need translations just type :
```
vosk_autosrt -S zh "Episode 1.mp4"
```

### Usage

```
usage: vosk_autosrt [-h] [-S SRC_LANGUAGE] [-D DST_LANGUAGE] [-lls] [-lld] [-F FORMAT] [-lf] [-C CONCURRENCY] [-v]
                       [source_path ...]

positional arguments:
  source_path           Path to the video or audio files to generate subtitles files (use wildcard for multiple files or separate
                        them with a space character e.g. "file 1.mp4" "file 2.mp4")

options:
  -h, --help            show this help message and exit
  -S SRC_LANGUAGE, --src-language SRC_LANGUAGE
                        Language code of the audio language spoken in video/audio source_path
  -D DST_LANGUAGE, --dst-language DST_LANGUAGE
                        Desired translation language code for the subtitles
  -lls, --list-src-languages
                        List all available source languages (vosk supported languages)
  -lld, --list-dst-languages
                        List all available destination languages (google translate supported languages)
  -F FORMAT, --format FORMAT
                        Desired subtitle format
  -lf, --list-formats   List all supported subtitle formats
  -C CONCURRENCY, --concurrency CONCURRENCY
                        Number of concurrent translate API requests to make
  -v, --version         show program's version number and exit```

### License

MIT

Check my other SPEECH RECOGNITIION + TRANSLATE PROJECTS https://github.com/botbahlul?tab=repositories
