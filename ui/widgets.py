from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from . import theme


class PillToggle(tk.Frame):
    """Pill-shaped on/off button.

    States are visually unambiguous:
        OFF -> dim panel background, faint label
        ON  -> accent fill, dark label, bold text
    Used for the overlay toggle row (Road Network / Coverage / Heatmap).
    """

    def __init__(self, master, label: str, initial: bool = True,
                 on_toggle: Optional[Callable[[bool], None]] = None,
                 width: int = 180, height: int = 34,
                 bg: str = theme.BG_BASE, **kwargs):
        super().__init__(master, bg=bg, highlightthickness=0, bd=0, **kwargs)
        self._label = label
        self._on_toggle = on_toggle
        self._state = bool(initial)
        self._hover = False
        self._bg = bg

        self._canvas = tk.Canvas(self, width=width, height=height,
                                  bg=bg, highlightthickness=0, bd=0)
        self._canvas.pack(fill="both", expand=True)

        self._pill_w = width
        self._pill_h = height
        self._draw()

        for evt, fn in [("<Enter>", self._on_enter), ("<Leave>", self._on_leave),
                        ("<Button-1>", self._on_click)]:
            self._canvas.bind(evt, fn)


    def set(self, value: bool) -> None:
        self._state = bool(value)
        self._draw()

    def get(self) -> bool:
        return self._state


    def _draw(self) -> None:
        self._canvas.delete("all")
        if self._state:
            fill = theme.ACCENT_BRIGHT if self._hover else theme.ACCENT
            text_color = theme.BG_BASE
            outline = theme.ACCENT_BRIGHT
            font = theme.FONT_BODY_BOLD
        else:
            fill = theme.BG_HOVER if self._hover else theme.BG_PANEL_ALT
            text_color = theme.TEXT_PRIMARY if self._hover else theme.TEXT_DIM
            outline = theme.ACCENT_DIM if self._hover else theme.BG_PANEL_DEEP
            font = theme.FONT_SMALL

        radius = self._pill_h // 2
        self._round_rect(2, 2, self._pill_w - 2, self._pill_h - 2,
                          radius=radius, fill=fill, outline=outline)
        self._canvas.create_text(self._pill_w // 2, self._pill_h // 2,
                                  text=self._label, fill=text_color,
                                  font=font)

    def _round_rect(self, x0, y0, x1, y1, radius=8, fill=theme.BG_PANEL,
                     outline=theme.ACCENT_DIM, width=1):
        points = [
            x0 + radius, y0,
            x1 - radius, y0,
            x1, y0,
            x1, y0 + radius,
            x1, y1 - radius,
            x1, y1,
            x1 - radius, y1,
            x0 + radius, y1,
            x0, y1,
            x0, y1 - radius,
            x0, y0 + radius,
            x0, y0,
        ]
        return self._canvas.create_polygon(points, smooth=True,
                                             fill=fill, outline=outline,
                                             width=width)


    def _on_enter(self, _event) -> None:
        self._hover = True
        self._canvas.configure(cursor="hand2")
        self._draw()

    def _on_leave(self, _event) -> None:
        self._hover = False
        self._canvas.configure(cursor="")
        self._draw()

    def _on_click(self, _event) -> None:
        self._state = not self._state
        self._draw()
        if self._on_toggle is not None:
            self._on_toggle(self._state)


class IconButton(tk.Frame):
    """Flat hover-aware button used in the left-nav. Always full-width."""

    def __init__(self, master, label: str, command: Callable,
                 active: bool = False, **kwargs):
        super().__init__(master, bg=theme.BG_PANEL, **kwargs)
        self._command = command
        self._active = active

        self._inner = tk.Frame(self, bg=theme.BG_PANEL,
                                highlightthickness=0, bd=0,
                                padx=14, pady=9)
        self._inner.pack(fill="x")

        self._label = tk.Label(self._inner, text=label,
                                bg=theme.BG_PANEL, fg=theme.TEXT_PRIMARY,
                                font=theme.FONT_BODY, anchor="w")
        self._label.pack(fill="x")
        self._set_visual()

        for w in (self, self._inner, self._label):
            w.bind("<Enter>", self._on_enter)
            w.bind("<Leave>", self._on_leave)
            w.bind("<Button-1>", self._on_click)

    def set_active(self, active: bool) -> None:
        self._active = active
        self._set_visual()

    def _set_visual(self) -> None:
        if self._active:
            for w in (self, self._inner):
                w.configure(bg=theme.BG_HOVER)
            self._label.configure(bg=theme.BG_HOVER, fg=theme.ACCENT,
                                    font=theme.FONT_BODY_BOLD)
        else:
            for w in (self, self._inner):
                w.configure(bg=theme.BG_PANEL)
            self._label.configure(bg=theme.BG_PANEL, fg=theme.TEXT_PRIMARY,
                                    font=theme.FONT_BODY)

    def _on_enter(self, _event) -> None:
        if self._active:
            return
        for w in (self, self._inner):
            w.configure(bg=theme.BG_HOVER)
        self._label.configure(bg=theme.BG_HOVER, fg=theme.ACCENT)
        self.configure(cursor="hand2")

    def _on_leave(self, _event) -> None:
        if self._active:
            return
        self._set_visual()
        self.configure(cursor="")

    def _on_click(self, _event) -> None:
        self._command()


class ScrollableFrame(tk.Frame):
    """A vertically scrollable Frame with a ttk-styled scrollbar.

    Use `.body` as the parent for child widgets; the wrapper handles the
    canvas + scrollbar plumbing.
    """

    def __init__(self, master, bg: str = theme.BG_PANEL, **kwargs):
        super().__init__(master, bg=bg, **kwargs)
        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self._scrollbar = ttk.Scrollbar(self, orient="vertical",
                                          style="Vertical.TScrollbar",
                                          command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self.body = tk.Frame(self._canvas, bg=bg)
        self._window = self._canvas.create_window((0, 0), window=self.body,
                                                    anchor="nw")
        self.body.bind("<Configure>", self._on_body_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        self.bind_all("<MouseWheel>", self._on_mousewheel, add=True)

    def scroll_to_bottom(self) -> None:
        self._canvas.update_idletasks()
        self._canvas.yview_moveto(1.0)

    def _on_body_configure(self, _event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)

    def _on_mousewheel(self, event) -> None:
        widget = self.winfo_containing(event.x_root, event.y_root)
        w = widget
        while w is not None:
            if w is self:
                self._canvas.yview_scroll(int(-event.delta / 60), "units")
                return
            w = w.master
