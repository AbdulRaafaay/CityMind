from __future__ import annotations

import math
import tkinter as tk
from typing import Callable, Dict, List, Optional, Set, Tuple

from . import theme
from ..city_graph import CityGraph, LETTER_CODE, LOC_EMPTY


CELL_PAD = 6
MIN_CELL = 44   # smallest cell size in pixels - keeps letters readable
COVERAGE_DASH = (5, 4)
ROUTE_DASH = (4, 3)


class CityGridCanvas(tk.Canvas):
    """Canvas that draws the city grid with toggleable overlays."""

    def __init__(self, master, graph: CityGraph,
                 on_cell_click: Optional[Callable[[int], None]] = None,
                 **kwargs):
        super().__init__(master, bg=theme.BG_BASE, highlightthickness=0,
                         bd=0, **kwargs)
        self.graph = graph
        self.on_cell_click = on_cell_click

        self.show_roads = True
        self.show_coverage = True
        self.show_heatmap = True

        self.ambulance_positions: List[int] = []
        self.civilians: List[int] = []
        self.medical_team_position: Optional[int] = None
        self.police_positions: List[int] = []
        self.recently_blocked: Set[Tuple[int, int]] = set()
        self.current_route: List[int] = []

        self._hover_cell: Optional[int] = None
        self._cell_size = MIN_CELL

        self.bind("<Configure>", lambda _event: self.redraw())
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)


    def set_overlay(self, name: str, enabled: bool) -> None:
        if name == "roads":
            self.show_roads = enabled
        elif name == "coverage":
            self.show_coverage = enabled
        elif name == "heatmap":
            self.show_heatmap = enabled
        elif name == "all":
            self.show_roads = enabled
            self.show_coverage = enabled
            self.show_heatmap = enabled
        self.redraw()

    def set_simulation_state(self, ambulances=None, civilians=None,
                             medical_team_position=None, police=None,
                             recently_blocked=None, current_route=None) -> None:
        if ambulances is not None:
            self.ambulance_positions = list(ambulances)
        if civilians is not None:
            self.civilians = list(civilians)
        if medical_team_position is not None:
            self.medical_team_position = medical_team_position
        if police is not None:
            self.police_positions = list(police)
        if recently_blocked is not None:
            self.recently_blocked = set(recently_blocked)
        if current_route is not None:
            self.current_route = list(current_route)
        self.redraw()


    def redraw(self) -> None:
        self.delete("all")
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        cell_w = (width - CELL_PAD * 2) / self.graph.cols
        cell_h = (height - CELL_PAD * 2) / self.graph.rows
        self._cell_size = max(MIN_CELL, int(min(cell_w, cell_h)))

        grid_w = self._cell_size * self.graph.cols
        grid_h = self._cell_size * self.graph.rows
        self._origin_x = (width - grid_w) // 2
        self._origin_y = (height - grid_h) // 2

        self._draw_cells()
        if self.show_roads:
            self._draw_roads()
        if self.show_coverage:
            self._draw_coverage()
        self._draw_route()
        self._draw_actors()

    def _draw_cells(self) -> None:
        for n in self.graph.all_nodes():
            x0, y0, x1, y1 = self._cell_bbox(n.row, n.col)
            fill = theme.TYPE_FILL.get(n.type, theme.BG_PANEL)
            border = theme.TYPE_BORDER.get(n.type, theme.ACCENT_DIM)
            if self.show_heatmap and n.type != LOC_EMPTY:
                fill = theme.RISK_TINT.get(n.crime_risk_level, fill)
            if self._hover_cell == n.id:
                fill = theme.BG_HOVER
                border = theme.ACCENT

            self._round_rect(x0 + 2, y0 + 2, x1 - 2, y1 - 2,
                             radius=8, fill=fill, outline=border, width=1)
            letter = LETTER_CODE.get(n.type, ".")
            self.create_text((x0 + x1) // 2, (y0 + y1) // 2,
                             text=letter, fill=theme.TEXT_PRIMARY,
                             font=theme.FONT_GRID_LETTER)

    def _draw_roads(self) -> None:
        for u, v, edge in self.graph.all_edges():
            ux, uy = self._cell_centre(u)
            vx, vy = self._cell_centre(v)
            if edge.blocked:
                if (u, v) in self.recently_blocked or (v, u) in self.recently_blocked:
                    colour = theme.DANGER
                    width = 2
                else:
                    colour = theme.TEXT_FAINT
                    width = 1
                self.create_line(ux, uy, vx, vy, fill=colour, width=width,
                                 dash=(4, 4))
            else:
                self.create_line(ux, uy, vx, vy, fill=theme.ACCENT_DIM, width=1)

    def _draw_coverage(self) -> None:
        radius = self._cell_size * 1.6
        for amb in self.ambulance_positions:
            cx, cy = self._cell_centre(amb)
            self.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                             outline=theme.ACCENT, dash=COVERAGE_DASH, width=1)

    def _draw_route(self) -> None:
        if len(self.current_route) < 2:
            return
        coords: List[float] = []
        for nid in self.current_route:
            x, y = self._cell_centre(nid)
            coords.extend((x, y))
        self.create_line(*coords, fill=theme.ALERT, dash=ROUTE_DASH, width=2,
                         smooth=False)

    def _draw_actors(self) -> None:
        for amb in self.ambulance_positions:
            cx, cy = self._cell_centre(amb)
            r = self._cell_size * 0.25
            self.create_rectangle(cx - r, cy - r, cx + r, cy + r,
                                  fill=theme.ALERT, outline="")
            self.create_text(cx, cy, text="A", fill=theme.BG_BASE,
                             font=theme.FONT_TINY)

        for civ in self.civilians:
            cx, cy = self._cell_centre(civ)
            r = self._cell_size * 0.25
            self.create_oval(cx - r, cy - r, cx + r, cy + r,
                             fill=theme.DANGER, outline=theme.TEXT_PRIMARY)

        if self.medical_team_position is not None:
            cx, cy = self._cell_centre(self.medical_team_position)
            r = self._cell_size * 0.35
            self.create_oval(cx - r, cy - r, cx + r, cy + r,
                             fill=theme.GOOD, outline=theme.TEXT_PRIMARY,
                             width=1)
            self.create_text(cx, cy, text="+", fill=theme.BG_BASE,
                             font=theme.FONT_BODY_BOLD)

        for officer in self.police_positions:
            cx, cy = self._cell_centre(officer)
            r = self._cell_size * 0.12
            self.create_oval(cx - r, cy - r, cx + r, cy + r,
                             fill=theme.ACCENT, outline="")


    def _cell_bbox(self, row: int, col: int) -> Tuple[int, int, int, int]:
        x0 = self._origin_x + col * self._cell_size
        y0 = self._origin_y + row * self._cell_size
        return x0, y0, x0 + self._cell_size, y0 + self._cell_size

    def _cell_centre(self, node_id: int) -> Tuple[float, float]:
        n = self.graph.node(node_id)
        x0, y0, x1, y1 = self._cell_bbox(n.row, n.col)
        return (x0 + x1) / 2, (y0 + y1) / 2

    def _round_rect(self, x0, y0, x1, y1, radius=8, **kwargs):
        """Draw a rounded rectangle - Tk has no native primitive."""
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
        return self.create_polygon(points, smooth=True, **kwargs)


    def _on_motion(self, event) -> None:
        node_id = self._cell_at(event.x, event.y)
        if node_id != self._hover_cell:
            self._hover_cell = node_id
            self.redraw()

    def _on_leave(self, _event) -> None:
        if self._hover_cell is not None:
            self._hover_cell = None
            self.redraw()

    def _on_click(self, event) -> None:
        node_id = self._cell_at(event.x, event.y)
        if node_id is not None and self.on_cell_click is not None:
            self.on_cell_click(node_id)

    def _cell_at(self, x: int, y: int) -> Optional[int]:
        col = (x - self._origin_x) // self._cell_size
        row = (y - self._origin_y) // self._cell_size
        if 0 <= row < self.graph.rows and 0 <= col < self.graph.cols:
            return self.graph.cell(row, col).id
        return None
