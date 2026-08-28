"""Standalone Tkinter GUI for tcgDataSynth Phase 7 generation.

Run with a normal desktop Python installation, not Blender's Text Editor:
    python gui.py
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from rules import combinations  # noqa: E402
from rules.generation import record_failed_seed, resume_next_index  # noqa: E402


_LIST_OPTIONS = {
    "layouts": ("table", "floating", "binder", "display_case", "hand", "stack"),
    "protections": ("none", "sleeve", "semi_rigid", "toploader", "slab"),
    "sleeve_types": ("clear", "opaque_back"),
    "sleeve_sizes": ("1mm", "2.5mm"),
    "finishes": ("normal", "holo"),
    "holo_regions": ("entire", "picture", "reverse"),
    "holo_patterns": ("none", "cosmos", "horizontal_lines", "water_web"),
    "damage": ("dirt", "scratches", "surface"),
    "binder_grids": ("1x1", "2x2", "3x3", "4x3"),
    "binder_contents": ("sleeved", "toploader", "slab"),
    "post_effects": combinations.POST_EFFECTS,
}
_LIGHTING_OPTIONS = ("spotlight", "point_lights", "occluders")


def _output_layout() -> config.OutputLayout:
    return config.OutputLayout(
        root=os.path.join(_ROOT, config.OUTPUT.root),
        images_subdir=config.OUTPUT.images_subdir,
        labels_subdir=config.OUTPUT.labels_subdir,
        yolo_labels_subdir=config.OUTPUT.yolo_labels_subdir,
        extra_labels_subdir=config.OUTPUT.extra_labels_subdir,
        manifest_name=config.OUTPUT.manifest_name,
    )


class GeneratorGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.config_path = os.path.join(_ROOT, config.CONFIG_FILENAME)
        self.worker_path = os.path.join(_ROOT, "blender", "generation_worker.py")
        self.output = _output_layout()
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.pause_requested = threading.Event()
        self.thread: threading.Thread | None = None
        self.process: subprocess.Popen[str] | None = None

        self.root.title("TCG Data Synth")
        self.root.geometry("820x760")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_widgets()
        self._load_config()
        self.root.after(100, self._drain_events)

    def _build_widgets(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(outer)
        top.pack(fill=tk.X)
        self.blender_executable = tk.StringVar()
        ttk.Label(top, text="Blender executable").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.blender_executable, width=74).grid(
            row=0, column=1, sticky="ew", padx=(8, 4))
        ttk.Button(top, text="Browse", command=self._browse_blender).grid(row=0, column=2)
        self.table_texture_dir = tk.StringVar()
        ttk.Label(top, text="Table texture directory").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.table_texture_dir, width=74).grid(
            row=1, column=1, sticky="ew", padx=(8, 4), pady=(4, 0))
        ttk.Button(top, text="Browse", command=self._browse_table_textures).grid(
            row=1, column=2, pady=(4, 0))
        top.columnconfigure(1, weight=1)

        basics = ttk.Frame(outer)
        basics.pack(fill=tk.X, pady=(8, 4))
        self.count = tk.StringVar()
        self.base_seed = tk.StringVar()
        self.back_probability = tk.StringVar()
        self.cardless_probability = tk.StringVar()
        self.physical_texture = tk.BooleanVar()
        self.export_yolo_segmentation = tk.BooleanVar()
        for column, (label, variable, width) in enumerate((
                ("Pairs", self.count, 8), ("Base seed", self.base_seed, 14),
                ("Back-facing probability", self.back_probability, 8),
                ("Cardless probability", self.cardless_probability, 8))):
            ttk.Label(basics, text=label).grid(row=0, column=column * 2, sticky="w")
            ttk.Entry(basics, textvariable=variable, width=width).grid(
                row=0, column=column * 2 + 1, sticky="w", padx=(4, 12))
        ttk.Checkbutton(basics, text="Physical holo texture", variable=self.physical_texture).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            basics, text="Export YOLO segmentation + extra labels",
            variable=self.export_yolo_segmentation).grid(
                row=1, column=3, columnspan=4, sticky="w", pady=(4, 0))

        canvas_frame = ttk.Frame(outer)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.controls = ttk.Frame(canvas)
        self.controls.bind("<Configure>", lambda _event: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.controls, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.checks: dict[tuple[str, str], tk.BooleanVar] = {}
        for group, values in _LIST_OPTIONS.items():
            box = ttk.LabelFrame(self.controls, text=group.replace("_", " ").title(), padding=6)
            box.pack(fill=tk.X, padx=2, pady=3)
            for index, value in enumerate(values):
                variable = tk.BooleanVar()
                self.checks[(group, value)] = variable
                ttk.Checkbutton(box, text=value.replace("_", " "), variable=variable).grid(
                    row=index // 4, column=index % 4, sticky="w", padx=(0, 16))
        lighting_box = ttk.LabelFrame(self.controls, text="Lighting", padding=6)
        lighting_box.pack(fill=tk.X, padx=2, pady=3)
        for index, value in enumerate(_LIGHTING_OPTIONS):
            variable = tk.BooleanVar()
            self.checks[("lighting", value)] = variable
            ttk.Checkbutton(lighting_box, text=value.replace("_", " "), variable=variable).grid(
                row=0, column=index, sticky="w", padx=(0, 16))

        actions = ttk.Frame(outer)
        actions.pack(fill=tk.X)
        self.start_button = ttk.Button(actions, text="Start / Resume", command=self._start)
        self.start_button.pack(side=tk.LEFT)
        self.pause_button = ttk.Button(actions, text="Pause", command=self._pause, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=6)
        self.save_button = ttk.Button(
            actions, text="Save Settings", command=self._save_settings)
        self.save_button.pack(side=tk.LEFT)
        self.reload_button = ttk.Button(
            actions, text="Reload Config", command=self._load_config)
        self.reload_button.pack(side=tk.LEFT, padx=6)
        self.status = tk.StringVar(value="Ready")
        ttk.Label(actions, textvariable=self.status).pack(side=tk.RIGHT)
        self.log = tk.Text(outer, height=8, state=tk.DISABLED, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, pady=(8, 0))

    def _browse_blender(self) -> None:
        path = filedialog.askopenfilename(title="Select blender.exe")
        if path:
            self.blender_executable.set(path)

    def _browse_table_textures(self) -> None:
        path = filedialog.askdirectory(title="Select table texture image directory")
        if path:
            self.table_texture_dir.set(path)

    def _load_config(self) -> None:
        if self._is_running():
            return
        settings = config.load_generation_settings(self.config_path)
        options = settings["enabled_options"]
        self.blender_executable.set(config.load_blender_executable(self.config_path))
        self.table_texture_dir.set(config.load_table_texture_dir(self.config_path))
        self.count.set(str(settings["count"]))
        self.base_seed.set(str(settings["base_seed"]))
        self.back_probability.set(str(options["back_to_camera_prob"]))
        self.cardless_probability.set(str(options["cardless_scene_prob"]))
        self.physical_texture.set(bool(options["physical_texture"]))
        self.export_yolo_segmentation.set(bool(settings["export_yolo_segmentation"]))
        for group, values in _LIST_OPTIONS.items():
            selected = set(options[group])
            for value in values:
                self.checks[(group, value)].set(value in selected)
        for value in _LIGHTING_OPTIONS:
            self.checks[("lighting", value)].set(bool(options["lighting"][value]))
        self.status.set("Loaded config.json")

    def _settings(self) -> dict[str, Any]:
        try:
            count = int(self.count.get())
            base_seed = int(self.base_seed.get())
            back_probability = float(self.back_probability.get())
            cardless_probability = float(self.cardless_probability.get())
        except ValueError as exc:
            raise ValueError(
                "Pairs, base seed, and both scene probabilities must be numeric") from exc
        options: dict[str, Any] = {
            group: [value for value in values if self.checks[(group, value)].get()]
            for group, values in _LIST_OPTIONS.items()
        }
        options["physical_texture"] = bool(self.physical_texture.get())
        options["lighting"] = {value: self.checks[("lighting", value)].get()
                               for value in _LIGHTING_OPTIONS}
        options["back_to_camera_prob"] = back_probability
        options["cardless_scene_prob"] = cardless_probability
        return {
            "count": count,
            "base_seed": base_seed,
            "export_yolo_segmentation": bool(self.export_yolo_segmentation.get()),
            "enabled_options": options,
        }

    def _save_settings(self) -> bool:
        try:
            settings = self._settings()
            config.save_generation_settings(settings, self.config_path)
            config.save_blender_executable(self.blender_executable.get(), self.config_path)
            config.save_table_texture_dir(self.table_texture_dir.get(), self.config_path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Invalid settings", str(exc), parent=self.root)
            return False
        self.status.set("Saved config.json")
        return True

    def _start(self) -> None:
        if self._is_running():
            return
        if not self._save_settings():
            return
        executable = self.blender_executable.get().strip()
        if not os.path.isfile(executable):
            messagebox.showerror("Blender not found", f"No executable at:\n{executable}", parent=self.root)
            return
        try:
            settings = config.load_generation_settings(self.config_path)
            next_index = resume_next_index(
                self.output, settings["base_seed"], settings["count"],
                require_yolo_segmentation=settings["export_yolo_segmentation"])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Cannot resume", str(exc), parent=self.root)
            return
        if next_index >= settings["count"]:
            self.status.set("Requested pair count is already complete")
            return
        self.pause_requested.clear()
        self.start_button.configure(state=tk.DISABLED)
        self.pause_button.configure(state=tk.NORMAL)
        self.save_button.configure(state=tk.DISABLED)
        self.reload_button.configure(state=tk.DISABLED)
        self.status.set(f"Starting index {next_index}")
        self.thread = threading.Thread(target=self._run_workers, args=(executable,), daemon=True)
        self.thread.start()

    def _pause(self) -> None:
        if not self._is_running():
            return
        self.pause_requested.set()
        self.pause_button.configure(state=tk.DISABLED)
        self.status.set("Pause requested; current Blender worker will finish")

    def _run_workers(self, executable: str) -> None:
        try:
            settings = config.load_generation_settings(self.config_path)
            while True:
                next_index = resume_next_index(
                    self.output, settings["base_seed"], settings["count"],
                    require_yolo_segmentation=settings["export_yolo_segmentation"])
                if next_index >= settings["count"]:
                    self.events.put(("finished", "Generation complete"))
                    return
                if self.pause_requested.is_set():
                    self.events.put(("finished", "Paused between completed pairs"))
                    return
                command = [executable, "-b", "--python-exit-code", "1",
                           "-P", self.worker_path, "--", "--index",
                           str(next_index), "--config", self.config_path,
                           "--output-root", self.output.root]
                self.events.put(("status", f"Rendering index {next_index}"))
                self.process = subprocess.Popen(
                    command, cwd=_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self.events.put(("log", line.rstrip()))
                return_code = self.process.wait()
                self.process = None
                failure = None
                if return_code != 0:
                    failure = f"Blender worker index {next_index} exited {return_code}"
                else:
                    advanced_index = resume_next_index(
                        self.output, settings["base_seed"], settings["count"],
                        require_yolo_segmentation=settings["export_yolo_segmentation"])
                    if advanced_index <= next_index:
                        failure = (
                            f"Blender worker index {next_index} exited successfully "
                            "without publishing output")
                if failure is not None:
                    outcome = record_failed_seed(
                        self.output, settings["base_seed"], settings["count"], next_index,
                        failure,
                        require_yolo_segmentation=settings["export_yolo_segmentation"])
                    if outcome == "recovered":
                        self.events.put((
                            "log", f"{failure}; recovered its published output and continuing"))
                    else:
                        seed = settings["base_seed"] + next_index
                        self.events.put((
                            "log", f"{failure}; skipped seed {seed} and continuing"))
                    continue
                self.events.put(("status", f"Completed index {next_index}"))
                if self.pause_requested.is_set():
                    self.events.put(("finished", "Paused after completed image/label pair"))
                    return
        except Exception as exc:  # noqa: BLE001
            self.events.put(("error", str(exc)))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self._append_log(value)
                elif kind == "status":
                    self.status.set(value)
                elif kind == "error":
                    self.status.set("Generation failed")
                    self._append_log(value)
                    messagebox.showerror("Generation failed", value, parent=self.root)
                    self._finish_run()
                elif kind == "finished":
                    self.status.set(value)
                    self._finish_run()
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _append_log(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _finish_run(self) -> None:
        self.process = None
        self.start_button.configure(state=tk.NORMAL)
        self.pause_button.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.NORMAL)
        self.reload_button.configure(state=tk.NORMAL)

    def _is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _on_close(self) -> None:
        if self._is_running():
            self._pause()
            messagebox.showinfo(
                "Pause requested",
                "The active Blender worker will finish its image/label pair. Close the GUI after "
                "the status changes to paused or complete.", parent=self.root)
            return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    GeneratorGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
