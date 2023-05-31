@echo off
setlocal

set "folderToDelete1=.\build"
set "folderToDelete2=.\dist"
set "fileToDelete1=.\vosk_autosrt.spec"

if exist "%folderToDelete1%" (
    rmdir /s /q "%folderToDelete1%"
    if errorlevel 1 (
        echo Error occurred while deleting the folder.
    )
)

if exist "%folderToDelete2%" (
    rmdir /s /q "%folderToDelete2%"
    if errorlevel 1 (
        echo Error occurred while deleting the folder.
    )
)

if exist "%fileToDelete1%" (
    del /s /q "%fileToDelete1%"
    if errorlevel 1 (
        echo Error occurred while deleting the file.
    )
)

pyinstaller ^
--add-data ".\libgcc_s_seh-1.dll;." ^
--add-data ".\libstdc++-6.dll;." ^
--add-data ".\libvosk.dll;." ^
--add-data ".\libwinpthread-1.dll;." ^
--hidden-import argparse ^
--hidden-import sounddevice ^
--hidden-import=sip --paths=/usr/local/lib/python3.8/site-packages/sipbuild ^
--additional-hooks-dir=./ ^
--onefile vosk_autosrt.py
