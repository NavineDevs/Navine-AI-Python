# Navine AI - Pure Python

Python-only product tree.

- Main hybrid product (Rust/C++/Go gateway/Docker): `C:\Users\hitbo\Downloads\Navine AI`
- This folder omits `native/`, `services/`, `deploy/` and compiled binaries.
- `data/`, `checkpoints/`, and `models/` are junctions to the main tree so weights are not duplicated.
- Chat / train / serve use the same Python package as main, without native accelerators (native_bridge falls back to pure Python).

Setup: create a venv, `pip install -r requirements.txt`, run the same scripts as main.
