#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

TITLE = 'Navine Multi-Tool'


def tool_calc() -> None:
    expr = input('Expression: ').strip()
    print('Result:', eval(expr, {'__builtins__': {}}, {}))


def tool_mkdir() -> None:
    path = Path(input('Folder path: ').strip())
    path.mkdir(parents=True, exist_ok=True)
    print('Created', path)


def tool_list() -> None:
    path = Path(input('Folder path: ').strip() or '.')
    for item in sorted(path.iterdir()):
        print(item.name)


def tool_ping() -> None:
    host = input('Host: ').strip()
    subprocess.run(['ping', '-n', '4', host], check=False)


def tool_sysinfo() -> None:
    print('Host:', socket.gethostname())
    print('User:', os.environ.get('USERNAME') or os.environ.get('USER'))
    print('CWD:', Path.cwd())


def tool_hash() -> None:
    path = Path(input('File path: ').strip())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print('SHA256:', digest)


def tool_copy() -> None:
    src = Path(input('Source: ').strip())
    dst = Path(input('Destination: ').strip())
    shutil.copy2(src, dst)
    print('Copied')


MENU = [
    ('Calculator', tool_calc),
    ('Create folder', tool_mkdir),
    ('List folder', tool_list),
    ('Ping host', tool_ping),
    ('System info', tool_sysinfo),
    ('SHA256 file', tool_hash),
    ('Copy file', tool_copy),
]


def main() -> None:
    while True:
        print('=' * 48)
        print(TITLE)
        print('=' * 48)
        for idx, (label, _) in enumerate(MENU, start=1):
            print(f'{idx}. {label}')
        print('0. Exit')
        choice = input('Select tool: ').strip()
        if choice == '0':
            print('Bye')
            return
        if choice.isdigit() and 1 <= int(choice) <= len(MENU):
            try:
                MENU[int(choice) - 1][1]()
            except Exception as exc:
                print('Error:', exc)
        else:
            print('Unknown option')


if __name__ == '__main__':
    main()
