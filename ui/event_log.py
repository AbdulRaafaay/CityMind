"""Live event log panel.

Each entry shows step number + a short message. Alerts (kind="flood",
"warning") are rendered with an amber bullet, success-y kinds (rescue,
finish) with green. The log auto-scrolls to the latest entry and supports a
short colour fade-in so the eye lands on the most recent event.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import List

from . import theme
from .widgets import ScrollableFrame
from ..simulation import SimulationEvent


ALERT_KINDS = {"flood", "warning"}
SUCCESS_KINDS = {"rescue", "finish"}


class EventLogPanel(tk.Frame):
    """Scrollable list of simulation events styled like an ops console."""

    def __init__(self, master, **kwargs):
        super().__init__(master, bg=theme.BG_PANEL,
                          highlightthickness=0, bd=0, **kwargs)

        header = ttk.Frame(self, style="Panel.TFrame",
                            padding=(14, 12, 14, 8))
        header.pack(fill="x")
        ttk.Label(header, text="EVENT LOG",
                  style="Caption.TLabel").pack(anchor="w")
        ttk.Label(header, text="Live simulation decisions",
                  style="Dim.TLabel").pack(anchor="w", pady=(2, 0))

        self._scroller = ScrollableFrame(self, bg=theme.BG_PANEL)
        self._scroller.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._entries: List[tk.Frame] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_event(self, event: SimulationEvent) -> None:
        bullet_colour = theme.ACCENT
        text_colour = theme.TEXT_DIM
        if event.kind in ALERT_KINDS:
            bullet_colour = theme.ALERT
            text_colour = theme.TEXT_PRIMARY
        elif event.kind in SUCCESS_KINDS:
            bullet_colour = theme.GOOD
            text_colour = theme.TEXT_PRIMARY

        row = tk.Frame(self._scroller.body, bg=theme.BG_PANEL_ALT,
                        highlightthickness=0, bd=0)
        row.pack(fill="x", padx=4, pady=2)

        # Coloured accent strip on the left edge.
        tk.Frame(row, bg=bullet_colour, width=3).pack(side="left", fill="y")

        inner = tk.Frame(row, bg=theme.BG_PANEL_ALT,
                          highlightthickness=0, bd=0,
                          padx=10, pady=8)
        inner.pack(side="left", fill="both", expand=True)

        head = tk.Frame(inner, bg=theme.BG_PANEL_ALT)
        head.pack(fill="x")
        step_text = f"STEP {event.step:02d}" if event.step > 0 else "INIT"
        tk.Label(head, text=step_text, bg=theme.BG_PANEL_ALT,
                  fg=bullet_colour, font=theme.FONT_TINY).pack(side="left")
        tk.Label(head, text=f"  ·  {event.kind}",
                  bg=theme.BG_PANEL_ALT, fg=theme.TEXT_FAINT,
                  font=theme.FONT_TINY).pack(side="left")

        body = tk.Label(inner, text=event.message, bg=theme.BG_PANEL_ALT,
                          fg=text_colour, font=theme.FONT_LOG, justify="left",
                          anchor="w", wraplength=280)
        body.pack(fill="x", pady=(3, 0))

        self._entries.append(row)
        self._scroller.scroll_to_bottom()

        # Subtle fade so a new entry feels alive without being noisy.
        self._fade_row(body, theme.ACCENT_BRIGHT, text_colour)

    def clear(self) -> None:
        for row in self._entries:
            row.destroy()
        self._entries.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fade_row(self, label: tk.Label, start_colour: str,
                   end_colour: str, step: int = 0) -> None:
        try:
            if not label.winfo_exists():
                return
            steps = 8
            if step >= steps:
                label.configure(fg=end_colour)
                return
            progress = step / steps
            label.configure(fg=_lerp_colour(start_colour, end_colour, progress))
            self.after(35, lambda: self._fade_row(label, start_colour,
                                                    end_colour, step + 1))
        except tk.TclError:
            return


def _lerp_colour(a: str, b: str, t: float) -> str:
    ar, ag, ab = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
    br, bg, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
    r = round(ar + (br - ar) * t)
    g = round(ag + (bg - ag) * t)
    b_ = round(ab + (bb - ab) * t)
    return f"#{r:02x}{g:02x}{b_:02x}"
