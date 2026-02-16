#!/usr/bin/env python3
"""
TNM111 Part 3

Controls:
- Left click a point: toggle "origin mode" (color by quadrant relative to that point)
- Right click a point: toggle "neighbors mode" (highlight 5 nearest neighbors)
- Keys:
    1 and 2 : switch between dataset
    r     : reset interactions


```bash
cd Assignment2
python3 scatterplot.py
```

"""

import csv
import math
import os
import tkinter as tk
from dataclasses import dataclass


# Configuring CSV paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_1_PATH = os.path.join(SCRIPT_DIR, "dataset1.csv")
DATASET_2_PATH = os.path.join(SCRIPT_DIR, "dataset2.csv")

@dataclass
class Point:
    x: float
    y: float
    cat: str
    # cached screen coords after mapping:
    sx: float = 0.0
    sy: float = 0.0

def read_points_from_csv(path):
    """
    Reads CSV with 3 columns: x, y, category.
    """
    # Load data from CSV file - expects numeric x and y columns plus category
    pts = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and len(row) >= 3:
                x = float(row[0])
                y = float(row[1])
                cat = row[2].strip()
                pts.append(Point(x=x, y=y, cat=cat))

    return pts

def nice_ticks(vmin, vmax, n=5):
    """Generate evenly spaced tick values for axis labels."""
    if n <= 1:
        return [vmin]
    # Create n evenly distributed ticks across the range
    step = (vmax - vmin) / (n - 1) if vmax != vmin else 1.0
    return [vmin + i * step for i in range(n)]



# ====== Scatterplot =======
class ScatterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Custom Scatter Plot (Tkinter)")

        self.canvas_w = 1000
        self.canvas_h = 700
        self.canvas = tk.Canvas(root, width=self.canvas_w, height=self.canvas_h, bg="white")
        self.canvas.pack(fill="both", expand=True)

        # Define margins around the plot area for axes and legend
        self.m_left = 80
        self.m_right = 250  # room for legend
        self.m_top = 50
        self.m_bottom = 70

        self.plot_w = self.canvas_w - self.m_left - self.m_right
        self.plot_h = self.canvas_h - self.m_top - self.m_bottom

        # Store loaded data and category mappings for visualization
        self.points = []
        self.categories = []
        self.cat_to_shape = {}
        self.shape_cycle = ["circle", "square", "triangle"]

        # Track the range of data values to properly scale points on screen
        self.xmin = 0.0
        self.xmax = 1.0
        self.ymin = 0.0
        self.ymax = 1.0

        # Keep track of user interactions (origin for quadrants, neighbors highlight)
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
        # Find min/max values in the dataset to determine axis ranges
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # Pad the ranges by 5% so data points don't sit right on the axis edges
        pad_x = (xmax - xmin) * 0.05
        pad_y = (ymax - ymin) * 0.05
        self.xmin, self.xmax = xmin - pad_x, xmax + pad_x
        self.ymin, self.ymax = ymin - pad_y, ymax + pad_y

    def data_to_screen(self, x, y):
        # Convert data coordinates to pixel positions on the canvas
        # First normalize to 0..1 based on data ranges
        nx = (x - self.xmin) / (self.xmax - self.xmin)
        ny = (y - self.ymin) / (self.ymax - self.ymin)
        # Then map to pixel coordinates, adding margins
        sx = self.m_left + nx * self.plot_w
        # Invert y because screen pixels increase downward, but data increases upward
        sy = self.m_top + (1.0 - ny) * self.plot_h
        return sx, sy

    def draw_axes(self):
        # Draw the x and y axes with tick marks and labels
        x0 = self.m_left
        y0 = self.m_top + self.plot_h
        x1 = self.m_left + self.plot_w
        y1 = self.m_top

        self.canvas.create_line(x0, y0, x1, y0, fill="black", width=2)  # x axis
        self.canvas.create_line(x0, y0, x0, y1, fill="black", width=2)  # y axis

        # Generate tick positions and draw them with labels
        xticks = nice_ticks(self.xmin, self.xmax, n=6)
        yticks = nice_ticks(self.ymin, self.ymax, n=6)

        # ticks on x-axis
        for t in xticks:
            sx, _ = self.data_to_screen(t, self.ymin)
            self.canvas.create_line(sx, y0, sx, y0 + 6, fill="black")
            self.canvas.create_text(sx, y0 + 20, text=f"{t:.2f}", fill="black", font=("Arial", 10))

        # ticks on y-axis
        for t in yticks:
            sx, sy = self.data_to_screen(self.xmin, t)
            self.canvas.create_line(x0 - 6, sy, x0, sy, fill="black")
            self.canvas.create_text(x0 - 35, sy, text=f"{t:.2f}", fill="black", font=("Arial", 10))

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
            text="Left click: origin/quadrants\nRight click: Highlight 5 nearest points\nKeys: 1 and 2 will change datasets",
            font=("Arial", 10), fill="#333"
        )

    def quadrant_color(self, p, origin):
        # Return a color based on which quadrant the point is in relative to the origin
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

        # Get the currently selected origin point if mode is active
        origin_pt = self.points[self.origin_idx] if (self.origin_active and self.origin_idx is not None) else None

        # Draw each point with appropriate styling based on current interaction modes
        for i, p in enumerate(self.points):
            p.sx, p.sy = self.data_to_screen(p.x, p.y)

            fill = "#666"
            outline = "black"
            width = 1

            # Apply quadrant coloring if origin mode is active
            if origin_pt is not None:
                fill = self.quadrant_color(p, origin_pt)

            # Highlight the selected point and its 5 nearest neighbors if neighbor mode is active
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

            # Make the origin point stand out with a thicker border
            if origin_pt is not None and i == self.origin_idx:
                outline = "black"
                width = 3

            shape = self.cat_to_shape.get(p.cat, "circle")
            self.draw_shape(p.sx, p.sy, shape, fill=fill, outline=outline, size=7, width=width)

        # Draw crosshairs through the origin point to show the quadrant boundaries
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

    # Redraws the entire canvas after interactions
    def redraw(self):
        self.canvas.delete("all")
        if not self.points:
            self.draw_message("No data loaded.")
            return
        self.draw_axes()
        self.draw_points()
        self.draw_legend()

    # Used to display messages
    def draw_message(self, msg):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas_w / 2, self.canvas_h / 2,
            text=msg, font=("Arial", 14), fill="#333", justify="center"
        )

    # ----- Interaction -----
    def find_nearest_point(self, sx, sy, max_px=12.0):
        # Find the closest point to a click, within a reasonable click radius
        if not self.points:
            return None
        best_i = None
        best_d = float("inf")
        for i, p in enumerate(self.points):
            # Calculate distance from click to point on screen
            dx = p.sx - sx
            dy = p.sy - sy
            d = math.hypot(dx, dy)
            if d < best_d:
                best_d = d
                best_i = i
        # Only return the point if the click was actually close to it
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
        # Find the k points closest to the selected point using Euclidean distance
        base = self.points[idx]
        dists = []
        for j, p in enumerate(self.points):
            if j == idx:
                continue
            # Calculate Euclidean distance in data space
            d = math.hypot(p.x - base.x, p.y - base.y)
            dists.append((d, j))
        # Sort by distance and return indices of k nearest neighbors
        dists.sort(key=lambda t: t[0])
        return [j for _, j in dists[:k]]

    def reset_modes(self):
        self.origin_active = False
        self.origin_idx = None
        self.neigh_active = False
        self.neigh_idx = None
        self.neigh_set = set()
        self.redraw()

    def build_category_shapes(self):
        cats = sorted({p.cat for p in self.points})
        self.categories = cats
        self.cat_to_shape = {}
        for i, c in enumerate(cats):
            self.cat_to_shape[c] = self.shape_cycle[i % len(self.shape_cycle)]

    def load_dataset(self, path):
        pts = read_points_from_csv(path)
        self.points = pts
        self.build_category_shapes()
        self.compute_ranges()
        self.reset_modes()
        self.redraw()
# ======= end of scatterplot =======



def main():
    root = tk.Tk()
    app = ScatterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
