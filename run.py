import sys
from pathlib import Path

ROOT = Path(r'C:\Users\hitbo\Downloads\Navine AI')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navine.llm import main_for

if __name__ == '__main__':
    raise SystemExit(main_for('text_code', sys.argv[1:]))
