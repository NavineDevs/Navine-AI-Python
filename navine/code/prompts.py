import re
from typing import Dict, Optional


LANGUAGE_TEMPLATES: Dict[str, str] = {
    "python": (
        "<code> language=python\n"
        "Write clean, idiomatic Python 3 code.\n"
        "Use type hints where appropriate.\n"
        "Task: {task}\n"
        "```python\n"
    ),
    "javascript": (
        "<code> language=javascript\n"
        "Write modern ES6+ JavaScript.\n"
        "Task: {task}\n"
        "```javascript\n"
    ),
    "typescript": (
        "<code> language=typescript\n"
        "Write typed TypeScript code.\n"
        "Task: {task}\n"
        "```typescript\n"
    ),
    "rust": (
        "<code> language=rust\n"
        "Write safe, idiomatic Rust.\n"
        "Task: {task}\n"
        "```rust\n"
    ),
    "go": (
        "<code> language=go\n"
        "Write idiomatic Go code.\n"
        "Task: {task}\n"
        "```go\n"
    ),
    "java": (
        "<code> language=java\n"
        "Write clean Java code.\n"
        "Task: {task}\n"
        "```java\n"
    ),
    "cpp": (
        "<code> language=cpp\n"
        "Write modern C++ code.\n"
        "Task: {task}\n"
        "```cpp\n"
    ),
    "csharp": (
        "<code> language=csharp\n"
        "Write clean, modern C# code.\n"
        "Task: {task}\n"
        "```csharp\n"
    ),
    "kotlin": (
        "<code> language=kotlin\n"
        "Write idiomatic Kotlin code.\n"
        "Task: {task}\n"
        "```kotlin\n"
    ),
    "swift": (
        "<code> language=swift\n"
        "Write idiomatic Swift code.\n"
        "Task: {task}\n"
        "```swift\n"
    ),
    "php": (
        "<code> language=php\n"
        "Write clean modern PHP code.\n"
        "Task: {task}\n"
        "```php\n"
    ),
    "ruby": (
        "<code> language=ruby\n"
        "Write idiomatic Ruby code.\n"
        "Task: {task}\n"
        "```ruby\n"
    ),
    "sql": (
        "<code> language=sql\n"
        "Write standard SQL.\n"
        "Task: {task}\n"
        "```sql\n"
    ),
    "bash": (
        "<code> language=bash\n"
        "Write a portable POSIX bash script.\n"
        "Task: {task}\n"
        "```bash\n"
    ),
    "lua": (
        "<code> language=lua\n"
        "Write clean Lua code.\n"
        "Task: {task}\n"
        "```lua\n"
    ),
    "scala": (
        "<code> language=scala\n"
        "Write idiomatic Scala code.\n"
        "Task: {task}\n"
        "```scala\n"
    ),
    "haskell": (
        "<code> language=haskell\n"
        "Write idiomatic Haskell code.\n"
        "Task: {task}\n"
        "```haskell\n"
    ),
    "r": (
        "<code> language=r\n"
        "Write clean R code.\n"
        "Task: {task}\n"
        "```r\n"
    ),
    "dart": (
        "<code> language=dart\n"
        "Write idiomatic Dart code.\n"
        "Task: {task}\n"
        "```dart\n"
    ),
    "elixir": (
        "<code> language=elixir\n"
        "Write idiomatic Elixir code.\n"
        "Task: {task}\n"
        "```elixir\n"
    ),
}

LANGUAGE_KEYWORDS = [
    ("batch", [r"\.bat\b", r"\bbatch\s+file\b", r"\bwindows\s+batch\b", r"\bcmd(?:\.exe)?\b", r"\bin\s+batch\b"]),
    ("typescript", [r"\btypescript\b", r"\bts\b", r"\btsx\b"]),
    ("javascript", [r"\bjavascript\b", r"\bjs\b", r"\bnode(?:\.?js)?\b", r"\breact\b"]),
    ("csharp", [r"\bc#", r"\bc\s*sharp\b", r"\bcsharp\b", r"\bdotnet\b", r"\basp\.net\b"]),
    ("cpp", [r"\bc\+\+\b", r"\bcpp\b"]),
    ("kotlin", [r"\bkotlin\b", r"\bandroid\b"]),
    ("swift", [r"\bswift\b", r"\bios\b", r"\bswiftui\b"]),
    ("rust", [r"\brust\b", r"\bcargo\b"]),
    ("go", [r"\bgolang\b", r"\bgo\b"]),
    ("java", [r"\bjava\b", r"\bspring\b"]),
    ("php", [r"\bphp\b", r"\blaravel\b"]),
    ("ruby", [r"\bruby\b", r"\brails\b"]),
    ("sql", [r"\bsql\b", r"\bquery\b", r"\bdatabase\b", r"\bselect\b.*\bfrom\b"]),
    ("bash", [r"\bbash\b", r"\bshell\s*script\b", r"\bshell\b"]),
    ("lua", [r"\blua\b", r"\bluvit\b"]),
    ("scala", [r"\bscala\b"]),
    ("haskell", [r"\bhaskell\b"]),
    ("r", [r"\br\s+language\b", r"\brlang\b", r"\brstudio\b", r"\b(?:in|using)\s+r\b"]),
    ("dart", [r"\bdart\b", r"\bflutter\b"]),
    ("elixir", [r"\belixir\b", r"\bphoenix\b"]),
    ("perl", [r"\bperl\b"]),
    ("matlab", [r"\bmatlab\b", r"\boctave\b"]),
    ("python", [r"\bpython\b", r"\bdjango\b", r"\bflask\b", r"\bpandas\b", r"\bnumpy\b"]),
]


def detect_language(task: str) -> str:
    task_lower = task.lower()
    if re.search(
        r"\b(?:tkinter|tk|pyqt(?:5|6)?|pyside(?:2|6)?|pygame|wxpython|customtkinter|dearpygui)\b",
        task_lower,
    ):
        return "python"
    if re.search(r"(?:\.bat\b|\bbatch(?:\s+file)?\b|\bwindows\s+batch\b|\bcmd(?:\.exe)?\b|\bin\s+batch\b)", task_lower):
        if not re.search(r"\bpython\b", task_lower):
            return "batch"
    if re.search(r"\b(?:gui|desktop\s+app|window)\b", task_lower) and not re.search(
        r"\b(?:javascript|typescript|java|rust|golang|c\+\+|cpp|c#|swift|kotlin|batch|\.bat)\b",
        task_lower,
    ):
        return "python"
    for lang, patterns in LANGUAGE_KEYWORDS:
        for pattern in patterns:
            if re.search(pattern, task_lower):
                return lang
    return "python"


def get_language_template(language: str) -> str:
    return LANGUAGE_TEMPLATES.get(language, LANGUAGE_TEMPLATES["python"])


def build_code_prompt(task: str, language: Optional[str] = None) -> str:
    lang = language or detect_language(task)
    template = get_language_template(lang)
    return template.format(task=task)
