#!/usr/bin/env python3
"""
TNM111 Part 3 — Custom scatterplot (Tkinter)

Controls:
- Left click a point: toggle "origin mode" (color by quadrant relative to that point)
- Right click a point: toggle "neighbors mode" (highlight 5 nearest neighbors)
- Keys:
    1 / 2 : load dataset 1 / 2 (edit file paths below)
    r     : reset interactions
"""

import csv
import math
import os
import tkinter as tk
from dataclasses import dataclass

# ----------------------------
# Configuring CSV paths
# ----------------------------
DATASET_1_PATH = "dataset1.csv"
DATASET_2_PATH = "dataset2.csv"

# ----------------------------
# Data model
# ----------------------------
@dataclass
class Point:
    x: float
    y: float
    cat: str
    # cached screen coords after mapping:
    sx: float = 0.0
    sy: float = 0.0

# ----------------------------
# Helpers
# ----------------------------
def try_float(s):
    try:
        return float(s)
    except Exception:
        return None

def read_points_from_csv(path):
    """
    Reads CSV with at least 2 numeric columns (x,y) and an optional category column.
    Accepts header or no header.
    If category missing, uses "default".
    """
    pts = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]

    if not rows:
        return pts

    # Detect header: if first row's first two cells are not floats -> header
    first = rows[0]
    header_like = True
    if len(first) >= 2:
        header_like = (try_float(first[0]) is None) or (try_float(first[1]) is None)

    start_idx = 1 if header_like else 0

    for row in rows[start_idx:]:
        if len(row) < 2:
            continue
        x = try_float(row[0])
        y = try_float(row[1])
        if x is None or y is None:
            continue
        cat = row[2].strip() if len(row) >= 3 and row[2].strip() else "default"
        pts.append(Point(x=x, y=y, cat=cat))

    return pts

def nice_ticks(vmin, vmax, n=5):
    """Simple linear ticks (intentionally basic)."""
    if n <= 1:
        return [vmin]
    step = (vmax - vmin) / (n - 1) if vmax != vmin else 1.0
    return [vmin + i * step for i in range(n)]

# ----------------------------
# Scatter plot app
# ----------------------------
class ScatterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Custom Scatter Plot (Tkinter)")

        self.canvas_w = 1000
        self.canvas_h = 700
        self.canvas = tk.Canvas(root, width=self.canvas_w, height=self.canvas_h, bg="white")
        self.canvas.pack(fill="both", expand=True)

        # Layout/margins
        self.m_left = 80
        self.m_right = 250  # room for legend
        self.m_top = 50
        self.m_bottom = 70

        self.plot_w = self.canvas_w - self.m_left - self.m_right
        self.plot_h = self.canvas_h - self.m_top - self.m_bottom

        # Data
        self.points = []
        self.categories = []
        self.cat_to_shape = {}
        self.shape_cycle = ["circle", "square", "triangle"]

        # Axis ranges
        self.xmin = 0.0
        self.xmax = 1.0
        self.ymin = 0.0
        self.ymax = 1.0

        # Interaction state
        self.origin_idx = None
        self.origin_active = False

        self.neigh_idx = None
        self.neigh_active = False
        self.neigh_set = set()

        # Bind events
        self.canvas.bind("<Button-1>", self.on_left_click)

        # Right-click variants (Mac + mouse)
        self.canvas.bind("<Button-3>", self.on_right_click)          # classic right click
        self.canvas.bind("<Button-2>", self.on_right_click)          # common on Mac trackpads
        self.canvas.bind("<Control-Button-1>", self.on_right_click)  # ctrl-click

        self.root.bind("r", lambda e: self.reset_modes())
        self.root.bind("1", lambda e: self.load_dataset(DATASET_1_PATH))
        self.root.bind("2", lambda e: self.load_dataset(DATASET_2_PATH))

        # Start with dataset 1 if available
        if os.path.exists(DATASET_1_PATH):
            self.load_dataset(DATASET_1_PATH)
        else:
            self.draw_message(f"Set DATASET_1_PATH / DATASET_2_PATH at top.\nMissing: {DATASET_1_PATH}")

    # ----- Core drawing -----
    def compute_ranges(self):
        if not self.points:
            self.xmin, self.xmax, self.ymin, self.ymax = 0, 1, 0, 1
            return

        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # Add padding so points aren't on border
        pad_x = (xmax - xmin) * 0.05 if xmax != xmin else 1.0
        pad_y = (ymax - ymin) * 0.05 if ymax != ymin else 1.0
        self.xmin, self.xmax = xmin - pad_x, xmax + pad_x
        self.ymin, self.ymax = ymin - pad_y, ymax + pad_y

    def data_to_screen(self, x, y):
        # normalize to 0..1
        nx = 0.5 if self.xmax == self.xmin else (x - self.xmin) / (self.xmax - self.xmin)
        ny = 0.5 if self.ymax == self.ymin else (y - self.ymin) / (self.ymax - self.ymin)
        sx = self.m_left + nx * self.plot_w
        sy = self.m_top + (1.0 - ny) * self.plot_h  # invert y for screen coords
        return sx, sy

    def draw_axes(self):
        # axes lines (left & bottom)
        x0 = self.m_left
        y0 = self.m_top + self.plot_h
        x1 = self.m_left + self.plot_w
        y1 = self.m_top

        self.canvas.create_line(x0, y0, x1, y0, fill="black", width=2)  # x axis
        self.canvas.create_line(x0, y0, x0, y1, fill="black", width=2)  # y axis

        # ticks
        xticks = nice_ticks(self.xmin, self.xmax, n=6)
        yticks = nice_ticks(self.ymin, self.ymax, n=6)

        # x ticks
        for t in xticks:
            sx, _ = self.data_to_screen(t, self.ymin)
            self.canvas.create_line(sx, y0, sx, y0 + 6, fill="black")
            self.canvas.create_text(sx, y0 + 20, text=f"{t:.2f}", fill="black", font=("Arial", 10))

        # y ticks
        for t in yticks:
            sx, sy = self.data_to_screen(self.xmin, t)
            self.canvas.create_line(x0 - 6, sy, x0, sy, fill="black")
            self.canvas.create_text(x0 - 35, sy, text=f"{t:.2f}", fill="black", font=("Arial", 10))

        # axis labels (optional but helpful)
        self.canvas.create_text((x0 + x1) / 2, y0 + 45, text="X", font=("Arial", 12, "bold"))
        self.canvas.create_text(x0 - 55, (y0 + y1) / 2, text="Y", font=("Arial", 12, "bold"), angle=90)

    def draw_legend(self):
        lx = self.m_left + self.plot_w + 30
        ly = self.m_top + 20
        self.canvas.create_text(lx, ly, text="Legend", anchor="w", font=("Arial", 13, "bold"))
        ly += 25

        for cat in self.categories:
            shape = self.cat_to_shape.get(cat, "circle")
            self.draw_shape(lx + 12, ly + 10, shape, fill="#666", outline="black", size=10)
            self.canvas.create_text(lx + 30, ly + 10, text=str(cat), anchor="w", font=("Arial", 11))
            ly += 28

        ly += 10
        self.canvas.create_text(
            lx, ly, anchor="w",
            text="Left click: origin/quadrants\nRight click: 5 nearest\nKeys: 1/2 load, r reset",
            font=("Arial", 10), fill="#333"
        )

    def quadrant_color(self, p, origin):
        # Q1: x>=ox,y>=oy ; Q2: x<ox,y>=oy ; Q3: x<ox,y<oy ; Q4: x>=ox,y<oy
        if p.x >= origin.x and p.y >= origin.y:
            return "#1f77b4"  # blue
        if p.x < origin.x and p.y >= origin.y:
            return "#ff7f0e"  # orange
        if p.x < origin.x and p.y < origin.y:
            return "#2ca02c"  # green
        return "#d62728"      # red

    def draw_points(self):
        if not self.points:
            return

        origin_pt = self.points[self.origin_idx] if (self.origin_active and self.origin_idx is not None) else None

        for i, p in enumerate(self.points):
            p.sx, p.sy = self.data_to_screen(p.x, p.y)

            fill = "#666"
            outline = "black"
            width = 1

            # quadrant mode coloring
            if origin_pt is not None:
                fill = self.quadrant_color(p, origin_pt)

            # neighbor highlight mode
            if self.neigh_active:
                if i == self.neigh_idx:
                    outline = "black"
                    width = 3
                elif i in self.neigh_set:
                    outline = "#9c27b0"
                    width = 3
                else:
                    outline = "#999"
                    width = 1

            # selected origin highlight
            if origin_pt is not None and i == self.origin_idx:
                outline = "black"
                width = 3

            shape = self.cat_to_shape.get(p.cat, "circle")
            self.draw_shape(p.sx, p.sy, shape, fill=fill, outline=outline, size=7, width=width)

        # crosshair for origin
        if origin_pt is not None:
            ox, oy = origin_pt.sx, origin_pt.sy
            self.canvas.create_line(ox, self.m_top, ox, self.m_top + self.plot_h, fill="#000", dash=(4, 4))
            self.canvas.create_line(self.m_left, oy, self.m_left + self.plot_w, oy, fill="#000", dash=(4, 4))

    def draw_shape(self, cx, cy, shape, fill, outline, size=7, width=1):
        s = size
        if shape == "circle":
            self.canvas.create_oval(cx - s, cy - s, cx + s, cy + s, fill=fill, outline=outline, width=width)
        elif shape == "square":
            self.canvas.create_rectangle(cx - s, cy - s, cx + s, cy + s, fill=fill, outline=outline, width=width)
        elif shape == "triangle":
            pts = [cx, cy - s, cx - s, cy + s, cx + s, cy + s]
            self.canvas.create_polygon(pts, fill=fill, outline=outline, width=width)

        else:
            self.canvas.create_oval(cx - s, cy - s, cx + s, cy + s, fill=fill, outline=outline, width=width)

    def redraw(self):
        self.canvas.delete("all")
        if not self.points:
            self.draw_message("No data loaded.")
            return
        self.draw_axes()
        self.draw_points()
        self.draw_legend()

    def draw_message(self, msg):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas_w / 2, self.canvas_h / 2,
            text=msg, font=("Arial", 14), fill="#333", justify="center"
        )

    # ----- Interaction -----
    def find_nearest_point(self, sx, sy, max_px=12.0):
        if not self.points:
            return None
        best_i = None
        best_d = float("inf")
        for i, p in enumerate(self.points):
            dx = p.sx - sx
            dy = p.sy - sy
            d = math.hypot(dx, dy)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is not None and best_d <= max_px:
            return best_i
        return None

    def on_left_click(self, event):
        idx = self.find_nearest_point(event.x, event.y)
        if idx is None:
            return

        if self.origin_active and self.origin_idx == idx:
            self.origin_active = False
            self.origin_idx = None
        else:
            self.origin_active = True
            self.origin_idx = idx

        self.redraw()

    def on_right_click(self, event):
        idx = self.find_nearest_point(event.x, event.y)
        if idx is None:
            return

        if self.neigh_active and self.neigh_idx == idx:
            self.neigh_active = False
            self.neigh_idx = None
            self.neigh_set = set()
        else:
            self.neigh_active = True
            self.neigh_idx = idx
            self.neigh_set = set(self.compute_k_nearest(idx, k=5))

        self.redraw()

    def compute_k_nearest(self, idx, k=5):
        base = self.points[idx]
        dists = []
        for j, p in enumerate(self.points):
            if j == idx:
                continue
            d = math.hypot(p.x - base.x, p.y - base.y)
            dists.append((d, j))
        dists.sort(key=lambda t: t[0])
        return [j for _, j in dists[:k]]

    def reset_modes(self):
        self.origin_active = False
        self.origin_idx = None
        self.neigh_active = False
        self.neigh_idx = None
        self.neigh_set = set()
        self.redraw()

    # ----- Data loading -----
    def build_category_shapes(self):
        cats = sorted({p.cat for p in self.points})
        self.categories = cats
        self.cat_to_shape = {}
        for i, c in enumerate(cats):
            self.cat_to_shape[c] = self.shape_cycle[i % len(self.shape_cycle)]

    def load_dataset(self, path):
        try:
            pts = read_points_from_csv(path)
        except Exception as e:
            self.points = []
            self.draw_message(f"Failed to load:\n{path}\n\n{e}")
            return

        if not pts:
            self.points = []
            self.draw_message(f"No points found in:\n{path}")
            return

        self.points = pts
        self.build_category_shapes()
        self.compute_ranges()
        self.reset_modes()
        self.redraw()

def main():
    root = tk.Tk()
    app = ScatterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
