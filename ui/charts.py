"""Equinor-styled Plotly charts and number formatting."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

NAVY = "#00243D"
RED = "#EB0037"
KARRY = "#FFE7D6"
PISTACHIO = "#9DBA00"
MOSS = "#007079"
MIST = "#D5EAF4"
GREY = "#6F6F6F"
LINE = "#DCDCDC"
SERIES = [MOSS, RED, NAVY, PISTACHIO, "#FF9200", "#7D4F9E", "#4BB4E6", GREY]
CATEGORY_COLOURS = {"Source": "#7D4F9E", "Charge": "#FF9200", "Reservoir": MOSS, "Trap": NAVY,
                    "Seal": PISTACHIO, "Other": GREY}


def fmt(x, digits: int = 3) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    if not np.isfinite(x):
        return "–"
    a = abs(x)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 1:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}g}"


def pct(x) -> str:
    try:
        return f"{100 * float(x):.1f} %"
    except (TypeError, ValueError):
        return "–"


def eq_layout(fig: go.Figure, title: str | None = None, height: int = 360, xtitle: str | None = None,
              ytitle: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=NAVY), x=0, xanchor="left") if title else None,
        height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Equinor, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif", size=12, color=NAVY),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", font=dict(color=NAVY)),
    )
    fig.update_xaxes(title=xtitle, gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    fig.update_yaxes(title=ytitle, gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    return fig


def histogram(x: np.ndarray, unit: str, title: str, colour: str = MOSS) -> go.Figure:
    x = np.asarray(x, float)
    fig = go.Figure(go.Histogram(x=x, nbinsx=60, marker_color=colour, opacity=0.85, name="Trials",
                                 hovertemplate="%{x}<br>%{y} trials<extra></extra>"))
    if x.size:
        q = np.quantile(x, [0.1, 0.5, 0.9])
        for val, name, dash in ((q[0], "P90", "dot"), (q[1], "P50", "dash"), (q[2], "P10", "dot")):
            fig.add_vline(x=val, line=dict(color=NAVY, dash=dash, width=1.5), annotation_text=name,
                          annotation_position="top", annotation_font=dict(size=10, color=NAVY))
        fig.add_vline(x=float(x.mean()), line=dict(color=RED, width=1.5), annotation_text="Mean",
                      annotation_position="top right", annotation_font=dict(size=10, color=RED))
    return eq_layout(fig, title, xtitle=unit, ytitle="Trials", legend=False)


def expectation_curves(curves: list[tuple[str, np.ndarray, float, str]], unit: str, title: str,
                       mefs: float | None = None, log_x: bool = False) -> go.Figure:
    """curves: (name, success-case volumes, chance multiplier, colour)."""
    fig = go.Figure()
    for name, vols, chance, colour in curves:
        v = np.sort(np.asarray(vols, float))
        if v.size == 0:
            continue
        exc = 1.0 - np.arange(v.size) / v.size
        if v.size > 400:
            idx = np.unique(np.linspace(0, v.size - 1, 400).astype(int))
            v, exc = v[idx], exc[idx]
        dash = "dash" if chance < 1 else "solid"
        fig.add_trace(go.Scatter(x=v, y=exc * chance, mode="lines", name=name, line=dict(color=colour, width=2, dash=dash),
                                 hovertemplate="%{x:.3g} " + unit + "<br>P(≥) %{y:.3f}<extra>" + name + "</extra>"))
    for p in (0.9, 0.5, 0.1):
        fig.add_hline(y=p, line=dict(color=LINE, width=1))
    if mefs:
        fig.add_vline(x=mefs, line=dict(color=RED, width=1.5, dash="dot"), annotation_text="MEFS",
                      annotation_font=dict(size=10, color=RED))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    if log_x:
        fig.update_xaxes(type="log")
    return eq_layout(fig, title, xtitle=unit, ytitle="Probability of exceeding")


def chance_bars(by_category: dict[str, float], scopes: dict[str, float], pg: float) -> go.Figure:
    cats = [c for c, v in by_category.items() if v < 1.0]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=cats, x=[by_category[c] for c in cats], orientation="h",
                         marker_color=[CATEGORY_COLOURS.get(c, GREY) for c in cats], name="Category chance",
                         text=[f"{by_category[c]:.2f}" for c in cats], textposition="outside",
                         hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    labels = list(scopes) + ["Pg"]
    fig.add_trace(go.Bar(y=labels, x=list(scopes.values()) + [pg], orientation="h",
                         marker_color=[MIST] * len(scopes) + [RED], name="By scope",
                         text=[f"{v:.3f}" for v in list(scopes.values()) + [pg]], textposition="outside",
                         hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.15])
    fig.update_yaxes(autorange="reversed")
    return eq_layout(fig, None, height=80 + 34 * (len(cats) + len(labels)), legend=False)


def tornado(rows: list[dict], base: float, unit: str, scale: float, title: str) -> go.Figure:
    rows = rows[:15][::-1]
    names = [r["Variable"] for r in rows]
    lo = [r["Output at low input"] / scale for r in rows]
    hi = [r["Output at high input"] / scale for r in rows]
    b = base / scale
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=[l - b for l in lo], base=b, orientation="h", name="Input at P90 (low)",
                         marker_color=MOSS, customdata=lo, hovertemplate="%{customdata:.3g}<extra>Low input</extra>"))
    fig.add_trace(go.Bar(y=names, x=[h - b for h in hi], base=b, orientation="h", name="Input at P10 (high)",
                         marker_color=RED, customdata=hi, hovertemplate="%{customdata:.3g}<extra>High input</extra>"))
    fig.add_vline(x=b, line=dict(color=NAVY, width=1.5))
    fig.update_layout(barmode="overlay")
    return eq_layout(fig, title, height=120 + 30 * len(rows), xtitle=unit)


def rank_bars(rows: list[dict], title: str) -> go.Figure:
    rows = rows[:15][::-1]
    fig = go.Figure(go.Bar(
        y=[r["Variable"] for r in rows], x=[100 * r["Contribution to variance"] for r in rows], orientation="h",
        marker_color=[MOSS if r["Rank correlation"] >= 0 else RED for r in rows],
        text=[f"ρ {r['Rank correlation']:+.2f}" for r in rows], textposition="outside",
        hovertemplate="%{y}<br>%{x:.1f} % of variance<extra></extra>"))
    return eq_layout(fig, title, height=120 + 30 * len(rows), xtitle="Contribution to variance (%, signed)", legend=False)


def area_depth_plot(tables: list[tuple[str, list[float], list[float], str]], contacts: dict[str, float],
                    area_unit: str, depth_unit: str) -> go.Figure:
    fig = go.Figure()
    for name, depth, area, colour in tables:
        fig.add_trace(go.Scatter(x=area, y=depth, mode="lines+markers", name=name, line=dict(color=colour, width=2)))
    for label, d in contacts.items():
        fig.add_hline(y=d, line=dict(color=RED, width=1, dash="dot"), annotation_text=label,
                      annotation_font=dict(size=10, color=RED), annotation_position="bottom right")
    fig.update_yaxes(autorange="reversed")
    return eq_layout(fig, None, height=320, xtitle=f"Area enclosed [{area_unit}]", ytitle=f"Depth [{depth_unit}]")


def count_bars(counts: np.ndarray, title: str) -> go.Figure:
    counts = np.asarray(counts, int)
    k = np.arange(counts.max() + 1) if counts.size else np.array([0])
    freq = np.bincount(counts, minlength=k.size) / max(counts.size, 1)
    fig = go.Figure(go.Bar(x=k, y=freq, marker_color=NAVY, text=[f"{f:.1%}" for f in freq], textposition="outside",
                           hovertemplate="%{x} discoveries: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(dtick=1)
    return eq_layout(fig, title, height=300, xtitle="Number of discoveries", ytitle="Probability", legend=False)


def phase_bars(labels: list[str], values: list[float]) -> go.Figure:
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=[PISTACHIO, RED, MOSS][:len(labels)],
                           text=[f"{v:.1%}" for v in values], textposition="outside"))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.1])
    return eq_layout(fig, "Sampled fluid phase", height=260, legend=False)


def dist_preview(samples: np.ndarray, colour: str = MOSS) -> go.Figure:
    fig = go.Figure(go.Histogram(x=samples, nbinsx=40, marker_color=colour, opacity=0.8))
    fig = eq_layout(fig, None, height=150, legend=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_yaxes(visible=False)
    return fig


# =============================================================================================
# Maps, sections, digitising
# =============================================================================================

FLUID = {"gas": "#E0413A", "oil": "#3B8C3A", "water": "#8DC3E8"}


def _runs(mask: np.ndarray) -> list[np.ndarray]:
    """Index arrays of consecutive True values."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    return np.split(idx, breaks + 1)


def section_figure(geom, length_unit: str, title: str | None = None, height: int = 380,
                   show_legend: bool = True) -> go.Figure:
    from prospectrisk.units import from_internal

    f = lambda a: from_internal(np.asarray(a, float), "length", length_unit)  # noqa: E731
    x = f(geom.x)
    fig = go.Figure()
    for name in ("water", "oil", "gas"):
        up, lo = (f(a) for a in geom.bands[name])
        shown = False
        for run in _runs(np.isfinite(up) & np.isfinite(lo)):
            if not np.any(lo[run] - up[run] > 0):
                continue
            fig.add_trace(go.Scatter(
                x=np.concatenate([x[run], x[run][::-1]]), y=np.concatenate([up[run], lo[run][::-1]]),
                fill="toself", mode="lines", line=dict(width=0), fillcolor=FLUID[name], name=name.capitalize(),
                legendgroup=name, showlegend=show_legend and not shown, hoverinfo="skip"))
            shown = True
    fig.add_trace(go.Scatter(x=x, y=f(geom.top), mode="lines", line=dict(color=NAVY, width=2), name="Top reservoir",
                             hovertemplate="%{x:.0f}, %{y:.1f}<extra>Top</extra>", showlegend=show_legend))
    fig.add_trace(go.Scatter(x=x, y=f(geom.base), mode="lines", line=dict(color=NAVY, width=1, dash="dash"),
                             name="Base reservoir", hovertemplate="%{x:.0f}, %{y:.1f}<extra>Base</extra>",
                             showlegend=show_legend))
    for name, z in geom.lines.items():
        if name == "Crest":
            continue
        fig.add_hline(y=float(f(z)), line=dict(color=GREY if name == "Spill" else NAVY, width=1, dash="dot"),
                      annotation_text=f"{name} {float(f(z)):.0f}", annotation_position="right",
                      annotation_font=dict(size=10, color=NAVY))
    fig.update_yaxes(autorange="reversed")
    return eq_layout(fig, title, height=height, xtitle=f"Distance [{length_unit}]", ytitle=f"Depth [{length_unit}]",
                     legend=show_legend)


def plan_figure(pv, xy_unit: str, z_unit: str, height: int = 460) -> go.Figure:
    g = pv.grid
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=g.x, y=g.y, z=g.z, colorscale=[[0, "#F7F4EC"], [1, "#9C8F76"]], reversescale=False,
        contours=dict(coloring="heatmap", showlabels=True, labelfont=dict(size=9, color=NAVY)),
        line=dict(width=0.6, color="#5B5446"), colorbar=dict(title=dict(text=f"Depth [{z_unit}]"), thickness=12),
        hovertemplate="x %{x:.0f}<br>y %{y:.0f}<br>depth %{z:.1f}<extra></extra>", name="Top structure"))
    fill = np.where(pv.gas, 1.0, np.where(pv.hc, 0.0, np.nan))
    if np.isfinite(fill).any():
        fig.add_trace(go.Heatmap(x=g.x, y=g.y, z=fill, zmin=0, zmax=1, showscale=False, opacity=0.6, hoverinfo="skip",
                                 colorscale=[[0, FLUID["oil"]], [0.5, FLUID["oil"]], [0.5, FLUID["gas"]],
                                             [1, FLUID["gas"]]], name="Hydrocarbons"))
    x1, y1, x2, y2 = pv.line
    fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode="lines+markers+text", text=["A", "A′"],
                             textposition="top center", line=dict(color=RED, width=2), marker=dict(size=7, color=RED),
                             name="Section line"))
    fig.add_trace(go.Scatter(x=[pv.crest_xy[0]], y=[pv.crest_xy[1]], mode="markers", name="Crest",
                             marker=dict(symbol="star", size=14, color=NAVY)))
    if pv.spill_xy is not None:
        fig.add_trace(go.Scatter(x=[pv.spill_xy[0]], y=[pv.spill_xy[1]], mode="markers", name="Spill point",
                                 marker=dict(symbol="x", size=12, color=RED, line=dict(width=2))))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return eq_layout(fig, None, height=height, xtitle=f"x [{xy_unit}]", ytitle=f"y [{xy_unit}]")


def digitise_figure(image, contours, current: list[tuple[float, float]], grid_n: int = 90,
                    height: int = 620) -> go.Figure:
    """Georeferenced image with a transparent click grid; saved contours and the current polyline on top."""
    fig = go.Figure()
    fig.add_layout_image(dict(source=image.data_uri, xref="x", yref="y", x=image.xmin, y=image.ymax,
                              sizex=image.xmax - image.xmin, sizey=image.ymax - image.ymin, sizing="stretch",
                              layer="below", opacity=1.0))
    ny = max(int(grid_n * (image.ymax - image.ymin) / (image.xmax - image.xmin) * image.width / image.height), 10)
    gx, gy = np.meshgrid(np.linspace(image.xmin, image.xmax, grid_n), np.linspace(image.ymin, image.ymax, ny))
    fig.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers", name="Click grid", showlegend=False,
                             marker=dict(size=7, color="rgba(0,0,0,0)"),
                             hovertemplate="x %{x:.1f}<br>y %{y:.1f}<extra>click to add</extra>"))
    for k, c in enumerate(contours):
        fig.add_trace(go.Scatter(x=c.x + c.x[:1], y=c.y + c.y[:1], mode="lines", name=f"{c.depth:g}",
                                 line=dict(color=SERIES[k % len(SERIES)], width=2),
                                 hovertemplate=f"contour {c.depth:g}<extra></extra>"))
    if current:
        cx, cy = zip(*current)
        fig.add_trace(go.Scatter(x=list(cx), y=list(cy), mode="lines+markers", name="Current contour",
                                 line=dict(color=RED, width=2, dash="dot"), marker=dict(size=8, color=RED)))
    fig.update_xaxes(range=[image.xmin, image.xmax], showgrid=False)
    fig.update_yaxes(range=[image.ymin, image.ymax], showgrid=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(dragmode="zoom", clickmode="event+select")
    return eq_layout(fig, None, height=height, xtitle="x", ytitle="y")


def area_depth_fill(depth: np.ndarray, area: np.ndarray, contact: float | None, goc: float | None,
                    area_unit: str, depth_unit: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=area, y=depth, mode="lines", line=dict(color=NAVY, width=2), name="Top structure"))
    for lo_z, hi_z, name in ((None, goc, "gas"), (goc, contact, "oil")):
        if hi_z is None:
            continue
        lo = depth.min() if lo_z is None else lo_z
        m = (depth >= lo) & (depth <= hi_z)
        if m.sum() >= 2:
            fig.add_trace(go.Scatter(x=np.concatenate([area[m], [0, 0]]), y=np.concatenate([depth[m], [hi_z, lo]]),
                                     fill="toself", mode="lines", line=dict(width=0), fillcolor=FLUID[name],
                                     name=name.capitalize(), hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    return eq_layout(fig, None, height=320, xtitle=f"Area [{area_unit}]", ytitle=f"Depth [{depth_unit}]")


# =============================================================================================
# Explore
# =============================================================================================

def scatter_xy(x: np.ndarray, y: np.ndarray, mask: np.ndarray, xlabel: str, ylabel: str,
               colour: np.ndarray | None = None, colour_label: str | None = None, trend=None,
               log_x: bool = False, log_y: bool = False, height: int = 460) -> go.Figure:
    fig = go.Figure()
    if colour is not None:
        fig.add_trace(go.Scattergl(x=x[mask], y=y[mask], mode="markers", name="Filtered in",
                                   marker=dict(size=4, color=colour[mask], colorscale="Viridis", showscale=True,
                                               colorbar=dict(title=dict(text=colour_label or ""), thickness=12)),
                                   hovertemplate="%{x:.4g}, %{y:.4g}<extra></extra>"))
    else:
        if (~mask).any():
            fig.add_trace(go.Scattergl(x=x[~mask], y=y[~mask], mode="markers", name="Filtered out",
                                       marker=dict(size=3, color="#C9CED3"), hovertemplate="%{x:.4g}, %{y:.4g}<extra></extra>"))
        fig.add_trace(go.Scattergl(x=x[mask], y=y[mask], mode="markers", name="Filtered in",
                                   marker=dict(size=4, color=MOSS, opacity=0.7),
                                   hovertemplate="%{x:.4g}, %{y:.4g}<extra></extra>"))
    if trend is not None:
        xm, ym, lo, hi = trend
        fig.add_trace(go.Scatter(x=np.concatenate([xm, xm[::-1]]), y=np.concatenate([hi, lo[::-1]]), fill="toself",
                                 mode="lines", line=dict(width=0), fillcolor="rgba(235,0,55,0.12)",
                                 name="P90–P10 band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xm, y=ym, mode="lines", line=dict(color=RED, width=2), name="Median trend"))
    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    fig.update_layout(dragmode="select")
    return eq_layout(fig, None, height=height, xtitle=xlabel, ytitle=ylabel)


def correlation_heatmap(m, height: int | None = None) -> go.Figure:
    labels = [str(c) for c in m.columns]
    z = m.to_numpy(float)
    text = [[("" if not np.isfinite(v) else f"{v:.2f}") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, zmin=-1, zmax=1, zmid=0, text=text, texttemplate="%{text}",
                               colorscale=[[0, RED], [0.5, "#FFFFFF"], [1, MOSS]],
                               colorbar=dict(title=dict(text="ρ"), thickness=12),
                               hovertemplate="%{y}<br>%{x}<br>ρ = %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-35)
    return eq_layout(fig, None, height=height or 200 + 34 * len(labels), legend=False)


def scatter_matrix(df, columns: list[str], mask: np.ndarray, short: dict[str, str]) -> go.Figure:
    dims = [dict(label=short.get(c, c), values=df[c].to_numpy(float)) for c in columns]
    colour = np.where(mask, 1.0, 0.0)
    fig = go.Figure(go.Splom(dimensions=dims, showupperhalf=False, diagonal=dict(visible=False),
                             marker=dict(size=2.5, color=colour, colorscale=[[0, "#C9CED3"], [1, MOSS]],
                                         showscale=False, opacity=0.6)))
    return eq_layout(fig, None, height=160 * len(columns) + 80, legend=False)


def overlay_histograms(series: list[tuple[str, np.ndarray, str]], xlabel: str, height: int = 340) -> go.Figure:
    fig = go.Figure()
    for name, v, colour in series:
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size:
            fig.add_trace(go.Histogram(x=v, name=name, marker_color=colour, opacity=0.6, nbinsx=60,
                                       histnorm="probability density"))
    fig.update_layout(barmode="overlay")
    return eq_layout(fig, None, height=height, xtitle=xlabel, ytitle="Density")


# =============================================================================================
# Recovery-factor benchmark
# =============================================================================================

def rf_benchmark_figure(curves_df, rf_band: tuple[float, float, float] | None,
                        k_band: tuple[float, float, float] | None, analogs=None, drive_ranges=None,
                        height: int = 460) -> go.Figure:
    fig = go.Figure()
    k = curves_df["Permeability [mD]"].to_numpy(float)
    if drive_ranges:
        for i, r in enumerate(drive_ranges):
            fig.add_hrect(y0=r["Low"], y1=r["High"], fillcolor=SERIES[(i + 3) % len(SERIES)], opacity=0.07,
                          line_width=0, annotation_text=r["Drive mechanism"], annotation_position="top left",
                          annotation_font=dict(size=9, color=GREY))
    if k_band is not None:
        fig.add_vrect(x0=k_band[0], x1=k_band[2], fillcolor=NAVY, opacity=0.08, line_width=0,
                      annotation_text="Permeability P90–P10", annotation_position="bottom left",
                      annotation_font=dict(size=9, color=NAVY))
    if rf_band is not None:
        fig.add_hrect(y0=rf_band[0], y1=rf_band[2], fillcolor=RED, opacity=0.10, line_width=0)
        fig.add_hline(y=rf_band[1], line=dict(color=RED, width=2), annotation_text="Input RF P50 (band P90–P10)",
                      annotation_position="bottom right", annotation_font=dict(size=10, color=RED))
    for i, col in enumerate([c for c in curves_df.columns if c != "Permeability [mD]"]):
        fig.add_trace(go.Scatter(x=k, y=curves_df[col], mode="lines", name=col,
                                 line=dict(color=SERIES[i % len(SERIES)], width=2.5),
                                 hovertemplate="k %{x:.3g} mD<br>RF %{y:.1%}<extra></extra>"))
    if analogs is not None and len(analogs):
        fig.add_trace(go.Scatter(x=analogs["Permeability [mD]"], y=analogs["RF"], mode="markers", name="Analogues",
                                 text=analogs.get("Name"), marker=dict(size=9, color=NAVY, symbol="diamond",
                                                                       line=dict(color="white", width=1)),
                                 hovertemplate="%{text}<br>k %{x:.3g} mD<br>RF %{y:.1%}<extra></extra>"))
    fig.update_xaxes(type="log")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return eq_layout(fig, None, height=height, xtitle="Permeability [mD]", ytitle="Recovery factor")


def rank_compare(labels: list[str], all_rho: list[float], filt_rho: list[float] | None, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=all_rho, orientation="h", name="All trials", marker_color=MOSS,
                         hovertemplate="%{y}<br>ρ %{x:.2f}<extra>All</extra>"))
    if filt_rho is not None:
        fig.add_trace(go.Bar(y=labels, x=filt_rho, orientation="h", name="Filtered in", marker_color=RED,
                             hovertemplate="%{y}<br>ρ %{x:.2f}<extra>Filtered</extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(autorange="reversed")
    return eq_layout(fig, title, height=140 + 34 * len(labels), xtitle="Spearman rank correlation")
