"""Shared by every panel: where things are, the palette, small widgets and file helpers."""

from __future__ import annotations

import ctypes
import json
import math
import sys
import tkinter as tk
from pathlib import Path


def project_root() -> Path:
    """The project folder: two levels above this file, or, for the packaged .exe
    (which runs from a temporary unpack folder), the nearest folder above the
    .exe that holds ``config/project.toml``."""
    if getattr(sys, "frozen", False):
        here = Path(sys.executable).resolve().parent
        for d in (here, *here.parents):
            if (d / "config" / "project.toml").is_file():
                return d
        return here
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solarflare.settings import load_settings  # noqa: E402

S = load_settings(str(ROOT / "config" / "project.toml"), root=str(ROOT))
OUTPUTS = S.outputs
JOBS = OUTPUTS / "jobs"
DOWNLOADS = S.data_root / "pradan1.issdc.gov.in" / "al1" / "protected" / "downloadData"
REFRESH_MS = 1000                          # the console reads the files every second
NO_WINDOW = 0x08000000                     # CREATE_NO_WINDOW
NEW_GROUP = 0x00000200                     # CREATE_NEW_PROCESS_GROUP
TEST_SUITES = ("test_correctness", "test_robustness", "test_scale", "test_extract", "test_hel1os",
               "test_forward", "test_goes", "test_physics", "test_catalog", "test_products", "test_pipeline")

# ---- palette: calm console, colour only where it carries state -------------------
GROUND = "#0c1215"
PANEL = "#121b20"
LINE = "#223038"
TEXT = "#cfd9dd"
MUTED = "#7d929b"
FAINT = "#4a5d66"
TEAL = "#46b3a0"      # training / primary series
STEEL = "#6d8fb3"     # secondary series, info
AMBER = "#d9a441"     # validating / warning / C alert
RED = "#d9665b"       # stalled / failed / M alert
GREEN = "#6cbf84"     # finished / ok
UI = ("Segoe UI", 9)
UI_B = ("Segoe UI Semibold", 9)
SMALL = ("Segoe UI", 8)
MONO = ("Consolas", 10)
MONO_BIG = ("Consolas", 15)


# ---- machine load (no extra dependency: Windows APIs through ctypes) -------------

class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _Filetime(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_ulong), ("hi", ctypes.c_ulong)]


def _ft(x) -> float:
    return (x.hi << 32) | x.lo


class CpuLoad:
    """Whole-machine CPU use between calls, from GetSystemTimes."""

    def __init__(self):
        self.prev = None

    def read(self) -> float | None:
        if sys.platform != "win32":
            return None
        idle, kernel, user = _Filetime(), _Filetime(), _Filetime()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel),
                                                     ctypes.byref(user)):
            return None
        now = (_ft(idle), _ft(kernel), _ft(user))
        prev, self.prev = self.prev, now
        if prev is None:
            return None
        di, dk, du = (n - p for n, p in zip(now, prev))
        busy = dk + du - di
        return 100.0 * busy / (dk + du) if (dk + du) > 0 else None


def memory() -> tuple[float, float] | None:
    """(used GB, total GB) of physical memory."""
    if sys.platform != "win32":
        return None
    m = _MemStatus()
    m.dwLength = ctypes.sizeof(_MemStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return None
    return (m.ullTotalPhys - m.ullAvailPhys) / 1e9, m.ullTotalPhys / 1e9


def read_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tail(p: Path, n: int = 24) -> list[str]:
    try:
        with Path(p).open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(size - 48000, 0))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return [ln.rstrip() for ln in text.replace("\r", "\n").split("\n") if ln.strip()][-n:]


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == 259                # STILL_ACTIVE


def fmt_dur(s: float | None) -> str:
    if s is None or not math.isfinite(s) or s < 0:
        return "--"
    s = int(s)
    h, m = divmod(s // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s % 60:02d}s"


def blend(c1: str, c2: str, t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


class Tile(tk.Frame):
    """A labelled number with a line of context and an optional progress bar."""

    def __init__(self, parent, label: str, bar: bool = False):
        super().__init__(parent, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        tk.Label(self, text=label.upper(), bg=PANEL, fg=MUTED, font=SMALL).pack(anchor="w", padx=12, pady=(6, 0))
        self.value = tk.Label(self, text="--", bg=PANEL, fg=TEXT, font=MONO_BIG)
        self.value.pack(anchor="w", padx=12)
        self.sub = tk.Label(self, text="", bg=PANEL, fg=MUTED, font=UI)
        self.sub.pack(anchor="w", padx=12, pady=(0, 3 if bar else 7))
        self.bar = None
        if bar:
            self.bar = tk.Canvas(self, height=4, bg=LINE, highlightthickness=0)
            self.bar.pack(fill="x", padx=12, pady=(0, 11))

    def set(self, value: str, sub: str = "", color: str = TEXT, frac: float | None = None):
        self.value.config(text=value, fg=color)
        self.sub.config(text=sub)
        if self.bar is not None:
            self.bar.delete("all")
            w = self.bar.winfo_width()
            if frac is not None and w > 1:
                self.bar.create_rectangle(0, 0, int(w * min(max(frac, 0), 1)), 4, fill=TEAL, width=0)


class Sparkline(tk.Frame):
    """A label, a live number and a small filled trace of the last values.

    Drawn on a canvas rather than with matplotlib: it is redrawn every second."""

    def __init__(self, parent, label: str, unit: str = "", span: int = 180, lo: float = 0.0,
                 hi: float | None = 100.0, color: str = TEAL, height: int = 34):
        super().__init__(parent, bg=PANEL)
        self.unit, self.span, self.lo, self.hi, self.color = unit, span, lo, hi, color
        self.values: list[float] = []
        top = tk.Frame(self, bg=PANEL)
        top.pack(fill="x")
        tk.Label(top, text=label.upper(), bg=PANEL, fg=MUTED, font=SMALL).pack(side="left")
        self.value = tk.Label(top, text="--", bg=PANEL, fg=TEXT, font=("Consolas", 9))
        self.value.pack(side="right")
        self.canvas = tk.Canvas(self, height=height, bg=PANEL, highlightthickness=0)
        self.canvas.pack(fill="x")

    def push(self, v: float | None, text: str | None = None) -> None:
        if v is not None and v == v:
            self.values.append(float(v))
            del self.values[:-self.span]
        self.value.config(text=text if text is not None else ("--" if v is None else f"{v:.0f}{self.unit}"),
                          fg=TEXT if v is not None else FAINT)
        self._draw()

    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 60), max(int(c["height"]), 10)
        if not self.values:
            return
        lo = self.lo if self.lo is not None else min(self.values)
        hi = self.hi if self.hi is not None else max(max(self.values), lo + 1e-9)
        if hi <= lo:
            hi = lo + 1e-9
        n = len(self.values)
        # while the history is short, spread it over the whole width rather than
        # leaving a bare strip: the trace gains resolution instead of crawling in
        step = w / max(min(n, self.span) - 1, 1) if n > 1 else w
        x0 = 0.0
        pts = []
        for i, v in enumerate(self.values):
            y = h - 2 - (h - 4) * min(max((v - lo) / (hi - lo), 0.0), 1.0)
            pts += [x0 + i * step, y]
        c.create_line(0, h - 2, w, h - 2, fill=LINE)
        if len(pts) >= 4:
            c.create_polygon([pts[0], h - 2] + pts + [pts[-2], h - 2], fill=blend(PANEL, self.color, 0.22),
                             outline="")
            c.create_line(pts, fill=self.color, width=1.4)
        elif pts:
            c.create_oval(pts[0] - 1, pts[1] - 1, pts[0] + 1, pts[1] + 1, fill=self.color, outline="")


def flat_button(parent, text, command, fg=TEXT):
    return tk.Button(parent, text=text, command=command, bg=PANEL, fg=fg, activebackground=LINE,
                     activeforeground=fg, relief="flat", bd=0, highlightthickness=1, highlightbackground=LINE,
                     font=UI, padx=12, pady=5, cursor="hand2", disabledforeground=FAINT)


def panel(parent, title: str) -> tk.Frame:
    """A bordered panel with a small uppercase caption; returns the frame."""
    f = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
    tk.Label(f, text=title, bg=PANEL, fg=MUTED, font=SMALL).pack(anchor="w", padx=12, pady=(9, 0))
    return f


def style_axes(ax, title: str = ""):
    ax.set_facecolor(PANEL)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=LINE, lw=0.6, alpha=0.8)
    if title:
        ax.set_title(title, loc="left", color=MUTED, fontsize=9, pad=8)


def muted_legend(ax, **kw):
    kw.setdefault("fontsize", 6.5)
    leg = ax.legend(frameon=False, **kw)
    for t in leg.get_texts():
        t.set_color(MUTED)
    return leg
