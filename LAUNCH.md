# Launch Navine AI - Python

1. Double-click `Navine AI - Python.bat` in this folder.
2. Click **Web UI** for chat, image, and video in the browser (`http://127.0.0.1:8766`).
3. Or use the desktop menu: Chat, Image, Video, Voice, Train.
4. Training:
   - GUI: **Train Everything** (full) or **Quick Train**
   - Command: `Navine AI - Python.bat train`

## Sync with other AIs

From `Navine AI`, run:

```text
python scripts/sync_all_six.py
```

This keeps code, data, checkpoints, configs, and model runners aligned across all six projects until you stop syncing.

## Modes

Chat modes: Auto, Chat, Code, Think, Detective, OSINT, Analyze. Image and video use prompt-only generation.

## First-time setup

```text
Navine AI - Python.bat setup
```

## Troubleshooting

- `Navine AI - Python.bat help`
- `python -m navine.cli doctor`
- Default text model: `text_enterprise`
- Port: `8766`
