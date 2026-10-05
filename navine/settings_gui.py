import tkinter as tk
from tkinter import messagebox, ttk

from navine.device_manager import detect_hardware, get_device_settings, save_device_settings
from navine.utils.generation import get_generation_path, save_generation_settings


def open_settings_window() -> None:
    root = tk.Tk()
    root.title("Navine AI - Python Settings")
    root.geometry("460x400")
    root.resizable(False, False)

    hw = detect_hardware()
    settings = get_device_settings()
    current_mode = str(settings.get("mode") or "auto").lower()
    current_gen = get_generation_path()

    frame = ttk.Frame(root, padding=16)
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(frame, text="Device mode", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
    mode_var = tk.StringVar(value=current_mode if current_mode in ("cpu", "gpu", "auto", "hybrid") else "auto")
    for label, value in (("Hybrid GPU+CPU", "hybrid"), ("Auto", "auto"), ("GPU", "gpu"), ("CPU", "cpu")):
        ttk.Radiobutton(frame, text=label, variable=mode_var, value=value).pack(anchor=tk.W, pady=2)

    ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
    ttk.Label(frame, text="Image / video generation", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
    gen_var = tk.StringVar(value="secondary" if current_gen == "secondary" else "primary")
    ttk.Radiobutton(
        frame,
        text="Primary (custom Navine checkpoint only)",
        variable=gen_var,
        value="primary",
    ).pack(anchor=tk.W, pady=2)
    ttk.Radiobutton(
        frame,
        text="Secondary (external HuggingFace diffusers teacher)",
        variable=gen_var,
        value="secondary",
    ).pack(anchor=tk.W, pady=2)

    ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
    ttk.Label(frame, text="Hardware", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
    lines = []
    if hw.get("cuda_available"):
        lines.append(f"CUDA: {hw.get('cuda_name')} ({hw.get('cuda_vram_gb', '?')} GB VRAM)")
    else:
        lines.append("CUDA: not available")
    if hw.get("mps_available"):
        lines.append("MPS: available")
    lines.append(f"CPU threads: {hw.get('cpu_count')}")
    for line in lines:
        ttk.Label(frame, text=line).pack(anchor=tk.W)

    status = ttk.Label(frame, text="")
    status.pack(anchor=tk.W, pady=8)

    def apply_settings() -> None:
        path = save_device_settings(mode_var.get())
        gen_path = save_generation_settings(gen_var.get())
        status.config(text=f"Saved device + generation ({gen_path}) to configs/navine.yaml")
        messagebox.showinfo("Navine AI - Python", "Settings saved.")

    def create_api_key() -> None:
        try:
            from navine.api.settings import get_key_store

            store = get_key_store()
            key_id, raw_key, entry = store.create("Navine AI - Python")
            messagebox.showinfo(
                "Navine AI - Python API Key",
                f"Key id: {key_id}\nPrefix: {entry.get('prefix')}\n\n{raw_key}\n\nSave this key now.",
            )
        except Exception as exc:
            messagebox.showerror("Navine AI - Python", str(exc))

    btn_row = ttk.Frame(frame)
    btn_row.pack(fill=tk.X, pady=8)
    ttk.Button(btn_row, text="Apply", command=apply_settings).pack(side=tk.LEFT, padx=(0, 8))
    ttk.Button(btn_row, text="Create API Key (Navine AI - Python)", command=create_api_key).pack(side=tk.LEFT)

    root.mainloop()
