from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = collect_submodules('sip')
datas = collect_data_files('sip')
