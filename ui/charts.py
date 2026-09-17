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
