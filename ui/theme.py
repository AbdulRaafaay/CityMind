from __future__ import annotations

import ctypes
import platform
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from typing import Dict



BG_BASE = "#0b1320"
BG_PANEL = "#131c2c"
BG_PANEL_ALT = "#1a2435"
BG_PANEL_DEEP = "#0f1726"
BG_HOVER = "#243349"
BG_INPUT = "#1d2a3e"

ACCENT = "#5fb1d9"
ACCENT_BRIGHT = "#7cc8ed"
ACCENT_DIM = "#3a6e87"
ALERT = "#f0a850"
DANGER = "#e26565"
GOOD = "#7cd180"

TEXT_PRIMARY = "#eef3fa"
TEXT_DIM = "#94a4bd"
TEXT_FAINT = "#5d6e85"

TYPE_FILL: Dict[str, str] = {
    "Residential": "#1a2940",
    "Hospital":    "#1f3a4a",
    "School":      "#1f3324",
    "Industrial":  "#3a2a18",
    "Power Plant": "#321f44",
    "Ambulance Depot": "#4a2424",
    "Empty":       "#101824",
}
TYPE_BORDER: Dict[str, str] = {
    "Residential": "#2c3f5a",
    "Hospital":    "#5fb1d9",
    "School":      "#7cd180",
    "Industrial":  "#f0a850",
    "Power Plant": "#bb7eea",
    "Ambulance Depot": "#e26565",
    "Empty":       "#1c2735",
}

RISK_TINT: Dict[str, str] = {
    "Low":    "#15301f",
    "Medium": "#3a2c14",
    "High":   "#4a1c20",
}



FONT_TITLE: tkfont.Font  # large heading
FONT_HEADING: tkfont.Font
FONT_BODY: tkfont.Font
FONT_BODY_BOLD: tkfont.Font
FONT_SMALL: tkfont.Font
FONT_TINY: tkfont.Font
FONT_GRID_LETTER: tkfont.Font
FONT_LOG: tkfont.Font


def setup_fonts(root: tk.Misc) -> None:
    """Build named tkfont objects after the Tk root is created."""
    global FONT_TITLE, FONT_HEADING, FONT_BODY, FONT_BODY_BOLD
    global FONT_SMALL, FONT_TINY, FONT_GRID_LETTER, FONT_LOG
    family = _pick_font(root, ["Segoe UI Variable", "Segoe UI", "Inter",
                                "Helvetica", "Arial"])
    mono = _pick_font(root, ["Cascadia Mono", "JetBrains Mono", "Consolas",
                              "Courier New"])
    FONT_TITLE = tkfont.Font(root=root, family=family, size=18, weight="bold")
    FONT_HEADING = tkfont.Font(root=root, family=family, size=12, weight="bold")
    FONT_BODY = tkfont.Font(root=root, family=family, size=10)
    FONT_BODY_BOLD = tkfont.Font(root=root, family=family, size=10, weight="bold")
    FONT_SMALL = tkfont.Font(root=root, family=family, size=9)
    FONT_TINY = tkfont.Font(root=root, family=family, size=8, weight="bold")
    FONT_GRID_LETTER = tkfont.Font(root=root, family=mono, size=15,
                                    weight="bold")
    FONT_LOG = tkfont.Font(root=root, family=mono, size=9)


def _pick_font(root: tk.Misc, candidates) -> str:
    """Return the first font family from `candidates` that's installed."""
    available = set(tkfont.families(root))
    for name in candidates:
        if name in available:
            return name
    return candidates[-1]



def enable_dpi_awareness() -> None:
    """Tell Windows we'll handle our own DPI scaling. Call before tk.Tk()."""
    if platform.system() != "Windows":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def apply_tk_scaling(root: tk.Tk) -> None:
    """Scale Tk's internal point size so widgets honour the current DPI."""
    try:
        dpi = root.winfo_fpixels("1i")
        root.tk.call("tk", "scaling", dpi / 72.0)
    except Exception:
        pass



def setup_styles(root: tk.Tk) -> ttk.Style:
    """Configure ttk widgets to match our dark theme."""
    style = ttk.Style(root)
    style.theme_use("clam")  # only ttk theme that respects custom backgrounds

    style.configure(".", background=BG_BASE, foreground=TEXT_PRIMARY,
                    font=FONT_BODY, borderwidth=0, relief="flat")

    style.configure("App.TFrame", background=BG_BASE)
    style.configure("Panel.TFrame", background=BG_PANEL)
    style.configure("PanelAlt.TFrame", background=BG_PANEL_ALT)
    style.configure("PanelDeep.TFrame", background=BG_PANEL_DEEP)
    style.configure("Card.TFrame", background=BG_PANEL_ALT)

    style.configure("App.TLabel", background=BG_BASE, foreground=TEXT_PRIMARY,
                    font=FONT_BODY)
    style.configure("Panel.TLabel", background=BG_PANEL, foreground=TEXT_PRIMARY,
                    font=FONT_BODY)
    style.configure("PanelAlt.TLabel", background=BG_PANEL_ALT,
                    foreground=TEXT_PRIMARY, font=FONT_BODY)
    style.configure("Title.TLabel", background=BG_BASE, foreground=TEXT_PRIMARY,
                    font=FONT_TITLE)
    style.configure("Heading.TLabel", background=BG_BASE,
                    foreground=TEXT_PRIMARY, font=FONT_HEADING)
    style.configure("Caption.TLabel", background=BG_PANEL,
                    foreground=TEXT_FAINT, font=FONT_TINY)
    style.configure("Caption.PanelAlt.TLabel", background=BG_PANEL_ALT,
                    foreground=TEXT_FAINT, font=FONT_TINY)
    style.configure("Dim.TLabel", background=BG_PANEL, foreground=TEXT_DIM,
                    font=FONT_SMALL)
    style.configure("Dim.PanelAlt.TLabel", background=BG_PANEL_ALT,
                    foreground=TEXT_DIM, font=FONT_SMALL)
    style.configure("Status.TLabel", background=BG_PANEL,
                    foreground=TEXT_PRIMARY, font=FONT_BODY_BOLD)
    style.configure("Brand.TLabel", background=BG_PANEL, foreground=ACCENT,
                    font=FONT_TITLE)
    style.configure("BrandSub.TLabel", background=BG_PANEL,
                    foreground=TEXT_DIM, font=FONT_SMALL)
    style.configure("Inspector.TLabel", background=BG_PANEL,
                    foreground=TEXT_PRIMARY, font=FONT_SMALL)

    style.configure("Nav.TButton", background=BG_PANEL,
                    foreground=TEXT_PRIMARY, font=FONT_BODY,
                    padding=(14, 8), anchor="w", borderwidth=0)
    style.map("Nav.TButton",
              background=[("active", BG_HOVER), ("pressed", BG_HOVER)],
              foreground=[("active", ACCENT), ("pressed", ACCENT)])

    style.configure("NavActive.TButton", background=BG_HOVER,
                    foreground=ACCENT, font=FONT_BODY_BOLD,
                    padding=(14, 8), anchor="w", borderwidth=0)
    style.map("NavActive.TButton",
              background=[("active", BG_HOVER), ("pressed", BG_HOVER)],
              foreground=[("active", ACCENT), ("pressed", ACCENT)])

    style.configure("Primary.TButton", background=ACCENT,
                    foreground=BG_BASE, font=FONT_BODY_BOLD,
                    padding=(16, 10), borderwidth=0)
    style.map("Primary.TButton",
              background=[("active", ACCENT_BRIGHT), ("pressed", ACCENT_DIM)])

    style.configure("Secondary.TButton", background=BG_PANEL_ALT,
                    foreground=TEXT_PRIMARY, font=FONT_BODY,
                    padding=(16, 10), borderwidth=0)
    style.map("Secondary.TButton",
              background=[("active", BG_HOVER), ("pressed", BG_HOVER)],
              foreground=[("active", ACCENT_BRIGHT)])

    style.configure("Speed.TButton", background=BG_PANEL_ALT,
                    foreground=TEXT_DIM, font=FONT_SMALL,
                    padding=(10, 6), borderwidth=0)
    style.map("Speed.TButton",
              background=[("active", BG_HOVER)],
              foreground=[("active", TEXT_PRIMARY)])
    style.configure("SpeedActive.TButton", background=ACCENT,
                    foreground=BG_BASE, font=FONT_BODY_BOLD,
                    padding=(10, 6), borderwidth=0)
    style.map("SpeedActive.TButton",
              background=[("active", ACCENT_BRIGHT)])

    style.configure("Vertical.TScrollbar",
                    background=BG_PANEL_ALT,
                    troughcolor=BG_PANEL,
                    arrowcolor=TEXT_DIM,
                    bordercolor=BG_PANEL,
                    lightcolor=BG_PANEL_ALT,
                    darkcolor=BG_PANEL_ALT,
                    gripcount=0,
                    borderwidth=0)
    style.map("Vertical.TScrollbar",
              background=[("active", BG_HOVER), ("pressed", ACCENT_DIM)])

    style.configure("TSeparator", background=BG_PANEL_DEEP)

    return style
