"""Structure maps: import, digitising support and area-depth derivation.

Supported sources
-----------------
* **Grids** — IRAP classic ASCII (RMS) and ZMAP+ (Petrel and most mapping packages).
* **Scattered points** — x, y, z text/CSV files, gridded by linear triangulation.
* **Contours** — depth, x, y (and optionally an id) per vertex, from a file or
  digitised on a georeferenced map image.

Area-depth derivation
---------------------
* From a grid: for each depth level the closure is the connected region of the
  surface shallower than that level that contains the crest. The **spill point**
  is found by a priority flood from the crest: the level at which the closure
  first crosses a saddle into another high, or reaches the map edge or undefined
  nodes. The saddle location is reported.
* From contours: the area inside each closed polygon (shoelace), summed when a
  depth has several polygons, with an explicit crest depth where the area is zero.

Depths are positive downwards. Map coordinates and depths keep their own units
(m or ft); conversion to the engine's internal acres/ft happens when an
:class:`~prospectrisk.volumetrics.AreaDepthTable` is produced.
"""
from __future__ import annotations

import base64
import io
import re
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator, griddata

from .units import ACRE_M2, FT_M
from .volumetrics import AreaDepthTable, VolumetricsError

IRAP_UNDEF = 9999900.0
LENGTH_UNITS = ("m", "ft")


class MapError(ValueError):
    pass


def _ft_per(unit: str) -> float:
    if unit not in LENGTH_UNITS:
        raise MapError(f"Map length unit must be 'm' or 'ft' (got '{unit}')")
    return 1.0 / FT_M if unit == "m" else 1.0


def _acres_per_sq(unit: str) -> float:
    return 1.0 / ACRE_M2 if unit == "m" else 1.0 / 43560.0


# =====================================================================================
# Grid
# =====================================================================================

@dataclass
class Grid:
    """Regular grid of node values; row 0 is the southern (minimum y) row."""

    x0: float
    y0: float
    dx: float
    dy: float
    z: np.ndarray                   # (ny, nx), NaN = undefined
    xy_unit: str = "m"
    z_unit: str = "m"
    rotation: float = 0.0           # degrees, informational (area is rotation invariant)
    name: str = "Surface"

    def __post_init__(self):
        self.z = np.asarray(self.z, float)
        if self.z.ndim != 2 or min(self.z.shape) < 3:
            raise MapError("A grid needs at least 3 × 3 nodes")
        if not (self.dx > 0 and self.dy > 0):
            raise MapError("Grid increments must be positive")
        if not np.isfinite(self.z).any():
            raise MapError("The grid has no defined values")
        _ft_per(self.xy_unit)
        _ft_per(self.z_unit)

    # ---- geometry ------------------------------------------------------------------
    @property
    def ny(self) -> int:
        return self.z.shape[0]

    @property
    def nx(self) -> int:
        return self.z.shape[1]

    @property
    def x(self) -> np.ndarray:
        return self.x0 + self.dx * np.arange(self.nx)

    @property
    def y(self) -> np.ndarray:
        return self.y0 + self.dy * np.arange(self.ny)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return float(self.x[0]), float(self.x[-1]), float(self.y[0]), float(self.y[-1])

    @property
    def cell_area_acres(self) -> float:
        return self.dx * self.dy * _acres_per_sq(self.xy_unit)

    def index_of(self, x: float, y: float) -> tuple[int, int]:
        i = int(np.clip(round((y - self.y0) / self.dy), 0, self.ny - 1))
        j = int(np.clip(round((x - self.x0) / self.dx), 0, self.nx - 1))
        return i, j

    def xy_of(self, i: int, j: int) -> tuple[float, float]:
        return float(self.x0 + j * self.dx), float(self.y0 + i * self.dy)

    def sample(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Bilinear interpolation; NaN outside the grid or next to undefined nodes."""
        f = RegularGridInterpolator((self.y, self.x), self.z, method="linear", bounds_error=False, fill_value=np.nan)
        return f(np.column_stack([np.asarray(ys, float), np.asarray(xs, float)]))

    def shallowest(self) -> tuple[int, int]:
        idx = int(np.nanargmin(self.z))
        return int(idx // self.nx), int(idx % self.nx)

    # ---- serialisation -----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        raw = np.asarray(self.z, np.float32).tobytes()
        return {"x0": self.x0, "y0": self.y0, "dx": self.dx, "dy": self.dy, "nx": self.nx, "ny": self.ny,
                "xy_unit": self.xy_unit, "z_unit": self.z_unit, "rotation": self.rotation, "name": self.name,
                "z_f32_zlib_b64": base64.b64encode(zlib.compress(raw, 6)).decode("ascii")}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Grid":
        raw = zlib.decompress(base64.b64decode(d["z_f32_zlib_b64"]))
        z = np.frombuffer(raw, np.float32).astype(float).reshape(int(d["ny"]), int(d["nx"]))
        return cls(float(d["x0"]), float(d["y0"]), float(d["dx"]), float(d["dy"]), z, d.get("xy_unit", "m"),
                   d.get("z_unit", "m"), float(d.get("rotation", 0.0)), d.get("name", "Surface"))


def _numbers(text: str) -> np.ndarray:
    toks = re.split(r"[\s,;]+", text.strip())
    try:
        return np.array([float(t) for t in toks if t], float)
    except ValueError as exc:
        raise MapError(f"Unexpected non-numeric value in grid data: {exc}") from exc


def _decode(data: str | bytes) -> str:
    if isinstance(data, bytes):
        for enc in ("utf-8", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
    return str(data)


def read_irap_ascii(data: str | bytes, xy_unit: str = "m", z_unit: str = "m", name: str = "Surface") -> Grid:
    """IRAP classic ASCII: ``-996 ny dx dy / xmin xmax ymin ymax / nx rot xori yori / 0 0 0 0 0 0 0`` + values."""
    v = _numbers(_decode(data))
    if v.size < 19 or int(v[0]) != -996:
        raise MapError("Not an IRAP classic ASCII grid (header must start with -996)")
    ny, dx, dy = int(v[1]), float(v[2]), float(v[3])
    nx, rot, xori, yori = int(v[8]), float(v[9]), float(v[10]), float(v[11])
    vals = v[19:]
    if vals.size < nx * ny:
        raise MapError(f"IRAP grid declares {nx} × {ny} nodes but contains {vals.size} values")
    z = vals[: nx * ny].reshape(ny, nx)
    z = np.where(np.abs(z) >= IRAP_UNDEF * 0.999, np.nan, z)
    return Grid(xori, yori, dx, dy, z, xy_unit, z_unit, rot, name)


def write_irap_ascii(g: Grid) -> str:
    z = np.where(np.isfinite(g.z), g.z, IRAP_UNDEF)
    x1, x2, y1, y2 = g.extent
    lines = [f"-996 {g.ny} {g.dx:.6f} {g.dy:.6f}", f"{x1:.6f} {x2:.6f} {y1:.6f} {y2:.6f}",
             f"{g.nx} {g.rotation:.6f} {g.x0:.6f} {g.y0:.6f}", "0 0 0 0 0 0 0"]
    flat = z.ravel()
    lines += [" ".join(f"{val:.4f}" for val in flat[i:i + 6]) for i in range(0, flat.size, 6)]
    return "\n".join(lines) + "\n"


def read_zmap(data: str | bytes, xy_unit: str = "m", z_unit: str = "m", name: str = "Surface") -> Grid:
    """ZMAP+ grid: header between '@' lines, values column by column starting at the top-left node."""
    lines = [ln for ln in _decode(data).splitlines() if not ln.lstrip().startswith("!")]
    at = [i for i, ln in enumerate(lines) if ln.strip().startswith("@")]
    if len(at) < 2:
        raise MapError("Not a ZMAP+ grid (missing '@' header lines)")
    head = [ln for ln in lines[at[0] + 1: at[1]] if ln.strip()]
    try:
        l1 = [t.strip() for t in head[0].split(",")]
        l2 = [t.strip() for t in head[1].split(",") if t.strip()]
        null_val = float(l1[1]) if l1[1] else float(l1[2])
        nrows, ncols = int(float(l2[0])), int(float(l2[1]))
        xmin, xmax, ymin, ymax = (float(v) for v in l2[2:6])
    except (IndexError, ValueError) as exc:
        raise MapError("Could not read the ZMAP+ header (expected field width, null value, rows, columns, "
                       "xmin, xmax, ymin, ymax)") from exc
    vals = _numbers(" ".join(lines[at[1] + 1:]))
    if vals.size < nrows * ncols:
        raise MapError(f"ZMAP+ grid declares {nrows} × {ncols} nodes but contains {vals.size} values")
    z = vals[: nrows * ncols].reshape(ncols, nrows).T[::-1, :]  # columns top->bottom  =>  row 0 = ymin
    tol = max(abs(null_val) * 1e-6, 1e-9)
    z = np.where(np.abs(z - null_val) <= tol, np.nan, z)
    dx = (xmax - xmin) / (ncols - 1)
    dy = (ymax - ymin) / (nrows - 1)
    return Grid(xmin, ymin, dx, dy, z, xy_unit, z_unit, 0.0, name)


def write_zmap(g: Grid, null_value: float = 1e30) -> str:
    x1, x2, y1, y2 = g.extent
    head = ["@SURFACE HEADER, GRID, 5", f"15, {null_value:.6E}, , 4, 1",
            f"{g.ny}, {g.nx}, {x1:.4f}, {x2:.4f}, {y1:.4f}, {y2:.4f}", "0.0, 0.0, 0.0", "@"]
    z = np.where(np.isfinite(g.z), g.z, null_value)[::-1, :].T.ravel()
    body = [" ".join(f"{v:15.4f}" if abs(v) < 1e20 else f"{v:15.6E}" for v in z[i:i + 5]) for i in range(0, z.size, 5)]
    return "\n".join(head + body) + "\n"


def read_xyz(data: str | bytes) -> np.ndarray:
    """First three numeric columns of every numeric line (comma, semicolon, tab or space separated)."""
    rows = []
    for ln in _decode(data).splitlines():
        toks = [t for t in re.split(r"[\s,;]+", ln.strip()) if t]
        try:
            nums = [float(t) for t in toks]
        except ValueError:
            continue
        if len(nums) >= 3:
            rows.append(nums[:3])
    if len(rows) < 4:
        raise MapError("Need at least four numeric x, y, z rows")
    return np.array(rows, float)


def grid_from_points(xyz: np.ndarray, cell: float | None = None, xy_unit: str = "m", z_unit: str = "m",
                     name: str = "Surface", max_nodes: int = 250) -> Grid:
    """Linear (Delaunay) gridding of scattered points; undefined outside the convex hull."""
    xyz = np.asarray(xyz, float)
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    if xyz.shape[0] < 4:
        raise MapError("Need at least four valid points to grid a surface")
    xmin, ymin = xyz[:, 0].min(), xyz[:, 1].min()
    xmax, ymax = xyz[:, 0].max(), xyz[:, 1].max()
    span = max(xmax - xmin, ymax - ymin)
    if span <= 0:
        raise MapError("Points have no horizontal extent")
    if cell is None or cell <= 0:
        cell = span / (max_nodes - 1)
    nx = int(np.floor((xmax - xmin) / cell)) + 1
    ny = int(np.floor((ymax - ymin) / cell)) + 1
    if nx * ny > 2_000_000:
        raise MapError(f"Cell size gives {nx} × {ny} nodes; use a larger cell size")
    gx, gy = np.meshgrid(xmin + cell * np.arange(nx), ymin + cell * np.arange(ny))
    try:
        z = griddata(xyz[:, :2], xyz[:, 2], (gx, gy), method="linear")
    except Exception as exc:  # scipy QhullError for collinear points
        raise MapError(f"Could not triangulate the points: {exc}") from exc
    return Grid(xmin, ymin, cell, cell, z, xy_unit, z_unit, 0.0, name)


# =====================================================================================
# Contours
# =====================================================================================

@dataclass
class Contour:
    depth: float
    x: list[float]
    y: list[float]

    def __post_init__(self):
        self.x = [float(v) for v in self.x]
        self.y = [float(v) for v in self.y]
        if len(self.x) != len(self.y):
            raise MapError("Contour x and y must have the same length")

    @property
    def n(self) -> int:
        return len(self.x)

    def area(self) -> float:
        """Enclosed area (map units²), polygon closed implicitly."""
        if self.n < 3:
            return 0.0
        x, y = np.asarray(self.x), np.asarray(self.y)
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def centroid(self) -> tuple[float, float]:
        return float(np.mean(self.x)), float(np.mean(self.y))

    def to_dict(self) -> dict[str, Any]:
        return {"depth": self.depth, "x": [round(v, 3) for v in self.x], "y": [round(v, 3) for v in self.y]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Contour":
        return cls(float(d["depth"]), d["x"], d["y"])


def read_contours(data: str | bytes) -> list[Contour]:
    """Contour vertices as rows of ``depth, x, y`` or ``id, depth, x, y`` (header row optional).

    With an id column, each id is one polygon. Without, a new polygon starts whenever
    the depth changes between consecutive rows.
    """
    text = _decode(data)
    rows, header = [], None
    for ln in text.splitlines():
        toks = [t for t in re.split(r"[\s,;]+", ln.strip()) if t]
        if not toks:
            continue
        try:
            rows.append([float(t) for t in toks])
        except ValueError:
            if header is None and not rows:
                header = [t.lower() for t in toks]
    if not rows:
        raise MapError("No numeric contour rows found")
    width = min(len(r) for r in rows)
    if width < 3:
        raise MapError("Contour rows need at least depth, x, y")
    arr = np.array([r[:width] for r in rows], float)
    if header and {"x", "y"} <= set(header):
        col = {n: i for i, n in enumerate(header)}
        dcol = next((col[n] for n in ("depth", "z", "tvdss", "value") if n in col), None)
        if dcol is None:
            raise MapError("Header needs a depth (or z) column")
        icol = next((col[n] for n in ("id", "polygon", "contour", "poly") if n in col), None)
        xs, ys, ds = arr[:, col["x"]], arr[:, col["y"]], arr[:, dcol]
        ids = arr[:, icol] if icol is not None else None
    elif width >= 4:
        ids, ds, xs, ys = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    else:
        ids, ds, xs, ys = None, arr[:, 0], arr[:, 1], arr[:, 2]
    out: list[Contour] = []
    if ids is not None:
        for pid in dict.fromkeys(ids.tolist()):
            m = ids == pid
            if len(set(ds[m].tolist())) > 1:
                raise MapError(f"Polygon {pid:g} has more than one depth")
            out.append(Contour(float(ds[m][0]), xs[m].tolist(), ys[m].tolist()))
    else:
        start = 0
        for i in range(1, len(ds) + 1):
            if i == len(ds) or ds[i] != ds[start]:
                out.append(Contour(float(ds[start]), xs[start:i].tolist(), ys[start:i].tolist()))
                start = i
    return [c for c in out if c.n >= 3]


def contour_area_depth(contours: list[Contour], crest_depth: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(depths, areas in map units², warnings). Areas are summed per depth and forced non-decreasing."""
    closed = [c for c in contours if c.n >= 3]
    if not closed:
        raise MapError("Need at least one polygon with three or more vertices")
    depths = sorted({c.depth for c in closed})
    if crest_depth >= depths[0]:
        raise MapError(f"Crest depth {crest_depth:g} must be shallower than the shallowest contour ({depths[0]:g})")
    areas = np.array([sum(c.area() for c in closed if c.depth == d) for d in depths])
    warnings = []
    multi = [d for d in depths if sum(1 for c in closed if c.depth == d) > 1]
    if multi:
        warnings.append("Several polygons share a depth (" + ", ".join(f"{d:g}" for d in multi) + "); their areas "
                        "are added. Remove polygons that belong to other closures.")
    fixed = np.maximum.accumulate(areas)
    if np.any(fixed > areas + 1e-9):
        bad = [f"{d:g}" for d, a, f in zip(depths, areas, fixed) if f > a + 1e-9]
        warnings.append("Area decreases with depth at " + ", ".join(bad) + "; those areas were raised to the "
                        "area above them. Check for open or mis-labelled contours.")
    return np.array([crest_depth] + depths), np.concatenate([[0.0], fixed]), warnings


def grid_from_contours(contours: list[Contour], crest: tuple[float, float, float] | None = None,
                       cell: float | None = None, xy_unit: str = "m", z_unit: str = "m") -> Grid:
    pts = [(x, y, c.depth) for c in contours for x, y in zip(c.x, c.y)]
    if crest is not None:
        pts.append(tuple(crest))
    return grid_from_points(np.array(pts, float), cell, xy_unit, z_unit, "Surface from contours")


# =====================================================================================
# Closure analysis on grids
# =====================================================================================

_STRUCT4 = ndimage.generate_binary_structure(2, 1)


def _boundary_mask(z: np.ndarray) -> np.ndarray:
    b = np.zeros(z.shape, bool)
    b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
    undefined = ~np.isfinite(z)
    return b | ndimage.binary_dilation(undefined, _STRUCT4)


def closure_region(z: np.ndarray, level: float, crest: tuple[int, int]) -> np.ndarray:
    """Connected region (4-neighbour) of nodes shallower than or at ``level`` containing the crest."""
    shallow = np.isfinite(z) & (z <= level)
    if not shallow[crest]:
        return np.zeros(z.shape, bool)
    lab, _ = ndimage.label(shallow, _STRUCT4)
    return lab == lab[crest]


@dataclass
class ClosureInfo:
    crest_depth: float
    crest_xy: tuple[float, float]
    spill_depth: float | None
    spill_xy: tuple[float, float] | None
    open_at_map_edge: bool
    warnings: list[str] = field(default_factory=list)


def find_spill(g: Grid, crest: tuple[int, int], min_relief: float | None = None) -> ClosureInfo:
    """Spill point of the closure around ``crest`` by priority flood.

    The flood rises from the crest, always taking the shallowest reachable node. The
    closure spills at the flood level where it either reaches the map edge / undefined
    nodes, or crosses a saddle into another high (a node shallower than the current
    flood level by more than ``min_relief``; smaller bumps are treated as noise).
    """
    import heapq

    z = g.z
    zc = float(z[crest])
    if not np.isfinite(zc):
        raise MapError("The crest node is undefined")
    zmax = float(np.nanmax(z))
    if min_relief is None:
        min_relief = 0.005 * max(zmax - float(np.nanmin(z)), 1e-9)
    boundary = _boundary_mask(z)
    warnings: list[str] = []
    if boundary[crest]:
        warnings.append("The crest lies on the map edge or next to undefined nodes; the closure is open.")
        return ClosureInfo(zc, g.xy_of(*crest), zc, g.xy_of(*crest), True, warnings)
    ny, nx = z.shape
    seen = np.zeros(z.shape, bool)
    parent = np.full(z.shape + (2,), -1, int)
    seen[crest] = True
    heap = [(zc, crest[0], crest[1])]
    level = zc

    def saddle(i: int, j: int) -> tuple[int, int]:
        """Deepest node on the flood path from the crest to (i, j)."""
        best, node = -np.inf, (i, j)
        while (i, j) != crest and i >= 0:
            if z[i, j] > best:
                best, node = float(z[i, j]), (i, j)
            i, j = int(parent[i, j, 0]), int(parent[i, j, 1])
        return node

    while heap:
        fill, i, j = heapq.heappop(heap)
        level = max(level, fill)
        if z[i, j] < level - min_relief:
            si, sj = saddle(i, j)
            return ClosureInfo(zc, g.xy_of(*crest), float(level), g.xy_of(si, sj), False,
                               warnings + ["Spills over a saddle into a neighbouring high."])
        if boundary[i, j]:
            return ClosureInfo(zc, g.xy_of(*crest), float(level), g.xy_of(i, j), True,
                               warnings + ["Spills at the map edge or into undefined nodes; the true spill point may "
                                           "lie outside the mapped area."])
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a_, b_ = i + di, j + dj
            if 0 <= a_ < ny and 0 <= b_ < nx and not seen[a_, b_] and np.isfinite(z[a_, b_]):
                seen[a_, b_] = True
                parent[a_, b_] = (i, j)
                heapq.heappush(heap, (max(fill, float(z[a_, b_])), a_, b_))
    warnings.append("No spill point found within the map.")
    return ClosureInfo(zc, g.xy_of(*crest), None, None, False, warnings)


def grid_area_depth(g: Grid, crest: tuple[int, int], base_depth: float, n_levels: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """(depths, areas in acres) from the crest to ``base_depth`` (map z units)."""
    zc = float(g.z[crest])
    if base_depth <= zc:
        raise MapError("The area-depth table must extend below the crest")
    levels = np.linspace(zc, base_depth, max(int(n_levels), 3))
    cell = g.cell_area_acres
    areas = np.array([closure_region(g.z, lv, crest).sum() * cell for lv in levels])
    areas[0] = 0.0
    return levels, np.maximum.accumulate(areas)


# =====================================================================================
# Image georeferencing (digitising)
# =====================================================================================

@dataclass
class MapImage:
    """Downscaled map image with the world coordinates of its edges."""

    png_b64: str
    width: int
    height: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def world(self, px: float, py: float) -> tuple[float, float]:
        """Pixel (column from left, row from top) to world coordinates."""
        return (self.xmin + px / self.width * (self.xmax - self.xmin),
                self.ymax - py / self.height * (self.ymax - self.ymin))

    def pixel(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.xmin) / (self.xmax - self.xmin) * self.width,
                (self.ymax - y) / (self.ymax - self.ymin) * self.height)

    @property
    def data_uri(self) -> str:
        return "data:image/png;base64," + self.png_b64

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("png_b64", "width", "height", "xmin", "xmax", "ymin", "ymax")}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MapImage":
        return cls(**{k: d[k] for k in ("png_b64", "width", "height", "xmin", "xmax", "ymin", "ymax")})


def load_image(data: bytes, xmin: float, xmax: float, ymin: float, ymax: float, max_px: int = 1400) -> MapImage:
    from PIL import Image

    if not (xmax > xmin and ymax > ymin):
        raise MapError("Image extent must have xmax > xmin and ymax > ymin")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:  # noqa: BLE001 - PIL raises many types
        raise MapError(f"Could not read the image: {exc}") from exc
    im = im.convert("RGB")
    scale = min(1.0, max_px / max(im.size))
    if scale < 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return MapImage(base64.b64encode(buf.getvalue()).decode("ascii"), im.width, im.height, xmin, xmax, ymin, ymax)


# =====================================================================================
# Structure map attached to a segment
# =====================================================================================

SOURCES = {"grid": "Imported grid", "points": "Gridded points", "contours": "Contours (file or digitised)"}


@dataclass
class StructureMap:
    source: str = "grid"
    top: Grid | None = None
    base: Grid | None = None
    contours: list[Contour] = field(default_factory=list)
    crest_depth: float | None = None           # contours: depth where area = 0
    crest_xy: tuple[float, float] | None = None
    image: MapImage | None = None
    xy_unit: str = "m"
    z_unit: str = "m"
    n_levels: int = 60
    section: list[float] | None = None         # x1, y1, x2, y2
    min_relief: float | None = None            # ignore highs with less relief (map z units); None = auto

    def __post_init__(self):
        self._sync_units()

    def _sync_units(self) -> None:
        """Grid-based maps take their units from the top grid (the grid is the source of truth)."""
        if self.source in ("grid", "points") and self.top is not None:
            self.xy_unit, self.z_unit = self.top.xy_unit, self.top.z_unit

    def validate(self) -> None:
        self._sync_units()
        if self.source not in SOURCES:
            raise MapError(f"Unknown map source '{self.source}'")
        _ft_per(self.xy_unit)
        _ft_per(self.z_unit)
        if self.source in ("grid", "points") and self.top is None:
            raise MapError("No top-structure grid loaded")
        if self.source == "contours":
            if not any(c.n >= 3 for c in self.contours):
                raise MapError("Digitise or load at least one closed contour")
            if self.crest_depth is None:
                raise MapError("Enter the crest depth for contour maps")

    # ---- derived surfaces --------------------------------------------------------------
    def top_grid(self) -> Grid | None:
        self._sync_units()
        if self.source in ("grid", "points"):
            return self.top
        good = [c for c in self.contours if c.n >= 3]
        if not good or self.crest_depth is None:
            return None
        cx, cy = self.crest_xy if self.crest_xy else min(good, key=lambda c: c.depth).centroid()
        try:
            return grid_from_contours(good, (cx, cy, self.crest_depth), None, self.xy_unit, self.z_unit)
        except MapError:
            return None

    def crest_index(self, g: Grid) -> tuple[int, int]:
        if self.crest_xy is not None:
            i, j = g.index_of(*self.crest_xy)
            if np.isfinite(g.z[i, j]):
                # climb to the local minimum so the crest is the true high point of that closure
                for _ in range(g.nx * g.ny):
                    win = g.z[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
                    if not np.isfinite(win).any():
                        break
                    k = np.nanargmin(win)
                    ni, nj = max(i - 1, 0) + k // win.shape[1], max(j - 1, 0) + k % win.shape[1]
                    if g.z[ni, nj] >= g.z[i, j]:
                        break
                    i, j = int(ni), int(nj)
                return i, j
        return g.shallowest()

    # ---- area-depth ----------------------------------------------------------------------
    def area_depth(self, spill_override: float | None = None) -> tuple[AreaDepthTable, AreaDepthTable | None, ClosureInfo]:
        """Top (and base) area-depth tables in internal units, and closure information in map z units."""
        self.validate()
        zf = _ft_per(self.z_unit)
        if self.source == "contours":
            d, a, warns = contour_area_depth(self.contours, float(self.crest_depth))
            a_ac = a * _acres_per_sq(self.xy_unit)
            deepest = max(c for c in d)
            good = [c for c in self.contours if c.n >= 3]
            cxy = self.crest_xy or min(good, key=lambda c: c.depth).centroid()
            info = ClosureInfo(float(self.crest_depth), cxy, float(deepest) if spill_override is None else spill_override,
                               None, False, warns + ["Contour maps: the deepest closed contour is used as the spill "
                                                     "depth unless you enter one."])
            try:
                top = AreaDepthTable(list(d * zf), list(a_ac))
            except VolumetricsError as exc:
                raise MapError(str(exc)) from exc
            return top, None, info
        g = self.top
        crest = self.crest_index(g)
        info = find_spill(g, crest, self.min_relief)
        if spill_override is not None:
            info.spill_depth = float(spill_override)
            info.warnings.append("Spill depth entered by the user.")
        base_level = info.spill_depth if info.spill_depth is not None else float(np.nanmax(g.z))
        if base_level <= info.crest_depth:
            raise MapError("Spill depth is at the crest: the structure has no closure")
        d, a = grid_area_depth(g, crest, base_level, self.n_levels)
        top = AreaDepthTable(list(d * zf), list(a))
        base_tab = None
        if self.base is not None:
            b = self.base
            if b.xy_unit != g.xy_unit or b.z_unit != g.z_unit:
                raise MapError("Top and base grids must use the same units")
            bi, bj = b.index_of(*g.xy_of(*crest))
            if not np.isfinite(b.z[bi, bj]):
                raise MapError("The base grid is undefined below the crest")
            bcrest = (bi, bj)
            bmax = base_level + (float(np.nanmax(b.z)) - float(np.nanmin(b.z)))
            bd, ba = grid_area_depth(b, bcrest, min(bmax, float(np.nanmax(b.z))), self.n_levels)
            base_tab = AreaDepthTable(list(bd * zf), list(ba))
        return top, base_tab, info

    # ---- serialisation -------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "xy_unit": self.xy_unit, "z_unit": self.z_unit, "n_levels": self.n_levels,
            "top": None if self.top is None else self.top.to_dict(),
            "base": None if self.base is None else self.base.to_dict(),
            "contours": [c.to_dict() for c in self.contours], "crest_depth": self.crest_depth,
            "crest_xy": None if self.crest_xy is None else list(self.crest_xy),
            "image": None if self.image is None else self.image.to_dict(),
            "section": None if self.section is None else list(self.section),
            "min_relief": self.min_relief,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "StructureMap | None":
        if not d:
            return None
        return cls(source=d.get("source", "grid"), xy_unit=d.get("xy_unit", "m"), z_unit=d.get("z_unit", "m"),
                   n_levels=int(d.get("n_levels", 60)),
                   top=Grid.from_dict(d["top"]) if d.get("top") else None,
                   base=Grid.from_dict(d["base"]) if d.get("base") else None,
                   contours=[Contour.from_dict(c) for c in d.get("contours") or []],
                   crest_depth=d.get("crest_depth"),
                   crest_xy=tuple(d["crest_xy"]) if d.get("crest_xy") else None,
                   image=MapImage.from_dict(d["image"]) if d.get("image") else None,
                   section=list(d["section"]) if d.get("section") else None,
                   min_relief=d.get("min_relief"))


def synthetic_dome(nx: int = 121, ny: int = 101, cell: float = 50.0, crest: float = 2000.0,
                   second_crest: float | None = 2100.0, noise: float = 0.0, seed: int = 1) -> Grid:
    """Two elliptical domes joined by a saddle — a demonstration and test surface (metres).

    Depth of each dome: crest + 300 × r², with r the elliptical distance (semi-axes 1400 m × 1000 m
    and 1000 m × 800 m). The combined surface is the shallower of the two, so the closure around the
    first crest spills over the saddle between them.
    """
    x = cell * np.arange(nx)
    y = cell * np.arange(ny)
    gx, gy = np.meshgrid(x, y)
    c1, c2, cy = x[nx // 3], x[int(nx * 0.8)], y[ny // 2]
    z = crest + 300.0 * (((gx - c1) / 1400.0) ** 2 + ((gy - cy) / 1000.0) ** 2)
    if second_crest is not None:
        z = np.minimum(z, second_crest + 300.0 * (((gx - c2) / 1000.0) ** 2 + ((gy - cy) / 800.0) ** 2))
    if noise:
        rng = np.random.default_rng(seed)
        z = z + ndimage.gaussian_filter(rng.normal(0, noise, z.shape), 2)
    return Grid(0.0, 0.0, cell, cell, z, "m", "m", 0.0, "Top reservoir (synthetic)")
