"""Recovery-factor benchmarks, mainly as a function of permeability.

Oil correlations (fractions of STOIIP)
--------------------------------------
* **API water drive** — Arps et al. (1967), API statistical study of water-drive
  sandstone reservoirs::

      RF = 0.54898 [phi (1 - Swi) / Boi]^0.0422 [k mu_wi / mu_oi]^0.0770 Swi^-0.1903 (pi / pa)^-0.2159

  with k in darcy, viscosities in cP. (Also published per acre-ft as
  4259 [phi (1 - Sw)/Boi]^1.0422 ..., the same equation multiplied by 7758 phi (1 - Sw)/Boi.)

* **API solution-gas drive** — Arps et al. (1967)::

      RF = 0.41815 [phi (1 - Sw) / Bob]^0.1611 [k / mu_ob]^0.0979 Sw^0.3722 (pb / pa)^0.1741

  with k in darcy, mu_ob (oil viscosity at bubble point) in cP.

* **Guthrie & Greenberger (1955)**, water-drive sandstones::

      RF = 0.2719 log10 k + 0.25569 Sw - 0.1355 log10 mu_o - 1.538 phi - 0.00035 h + 0.11403

  with k in mD, mu_o in cP, h (net pay) in ft.

These are regressions on US reservoir data from the 1950s-60s: use them as a
sanity check against analogues, not as a prediction.

Gas
---
* Volumetric depletion (p/z):  RF = 1 - (pa / za) / (pi / zi).
* Water drive with trapped gas, volumetric sweep Ev::

      RF = 1 - [Ev Sgr / Sgi + (1 - Ev)] (Bgi / Bga),   Bgi / Bga = (zi pa) / (za pi)

Typical ranges by drive mechanism (AAPG Wiki, "Drive mechanisms and recovery"):
solution gas 5-30 %, gas cap 20-40 %, water drive 35-75 %, gravity drainage
5-30 % incremental.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DRIVE_RANGES = [
    {"Drive mechanism": "Solution gas drive", "Low": 0.05, "High": 0.30, "Fluid": "oil"},
    {"Drive mechanism": "Gas cap drive", "Low": 0.20, "High": 0.40, "Fluid": "oil"},
    {"Drive mechanism": "Water drive", "Low": 0.35, "High": 0.75, "Fluid": "oil"},
    {"Drive mechanism": "Gravity drainage (incremental)", "Low": 0.05, "High": 0.30, "Fluid": "oil"},
]
DRIVE_RANGES_SOURCE = "AAPG Wiki, Drive mechanisms and recovery (typical ranges, % of OOIP)"

CORRELATIONS = {
    "api_water": "API water drive (Arps et al. 1967)",
    "api_solution_gas": "API solution-gas drive (Arps et al. 1967)",
    "guthrie_greenberger": "Guthrie & Greenberger (1955), water drive",
}


class RecoveryError(ValueError):
    pass


def _pos(name: str, v) -> np.ndarray:
    a = np.asarray(v, float)
    if np.any(~np.isfinite(a)) or np.any(a <= 0):
        raise RecoveryError(f"{name} must be positive")
    return a


def _fraction(name: str, v) -> np.ndarray:
    a = np.asarray(v, float)
    if np.any(~np.isfinite(a)) or np.any((a <= 0) | (a >= 1)):
        raise RecoveryError(f"{name} must be a fraction between 0 and 1")
    return a


def api_water_drive(phi, sw, boi, k_md, mu_w, mu_o, p_i, p_a) -> np.ndarray:
    phi, sw, boi, k = _fraction("Porosity", phi), _fraction("Water saturation", sw), \
        _pos("Bo", boi), _pos("Permeability", k_md)
    mu_w, mu_o, p_i, p_a = _pos("Water viscosity", mu_w), _pos("Oil viscosity", mu_o), _pos("Initial pressure", p_i), \
        _pos("Abandonment pressure", p_a)
    rf = 0.54898 * (phi * (1 - sw) / boi) ** 0.0422 * (k / 1000.0 * mu_w / mu_o) ** 0.0770 * sw ** -0.1903 \
        * (p_i / p_a) ** -0.2159
    return np.clip(rf, 0.0, 1.0)


def api_solution_gas(phi, sw, bob, k_md, mu_ob, p_b, p_a) -> np.ndarray:
    phi, sw, bob, k = _fraction("Porosity", phi), _fraction("Water saturation", sw), \
        _pos("Bo", bob), _pos("Permeability", k_md)
    mu_ob, p_b, p_a = _pos("Oil viscosity", mu_ob), _pos("Bubble-point pressure", p_b), _pos("Abandonment pressure", p_a)
    rf = 0.41815 * (phi * (1 - sw) / bob) ** 0.1611 * (k / 1000.0 / mu_ob) ** 0.0979 * sw ** 0.3722 \
        * (p_b / p_a) ** 0.1741
    return np.clip(rf, 0.0, 1.0)


def guthrie_greenberger(k_md, sw, mu_o, phi, h_ft) -> np.ndarray:
    k, mu_o = _pos("Permeability", k_md), _pos("Oil viscosity", mu_o)
    sw, phi = _fraction("Water saturation", sw), _fraction("Porosity", phi)
    rf = 0.2719 * np.log10(k) + 0.25569 * sw - 0.1355 * np.log10(mu_o) \
        - 1.538 * phi - 0.00035 * np.asarray(h_ft, float) + 0.11403
    return np.clip(rf, 0.0, 1.0)


def gas_depletion(p_i, z_i, p_a, z_a) -> np.ndarray:
    p_i, z_i, p_a, z_a = _pos("Initial pressure", p_i), _pos("zi", z_i), _pos("Abandonment pressure", p_a), _pos("za", z_a)
    return np.clip(1.0 - (p_a / z_a) / (p_i / z_i), 0.0, 1.0)


def gas_water_drive(p_i, z_i, p_a, z_a, sgi, sgr, sweep=1.0) -> np.ndarray:
    sgi, sgr = _pos("Initial gas saturation", sgi), np.asarray(sgr, float)
    bratio = (np.asarray(z_i, float) * np.asarray(p_a, float)) / (np.asarray(z_a, float) * np.asarray(p_i, float))
    return np.clip(1.0 - (sweep * sgr / sgi + (1.0 - sweep)) * bratio, 0.0, 1.0)


@dataclass
class OilBenchmarkInputs:
    """Reservoir and fluid parameters for the oil correlations (field units: psia, cP, ft)."""

    porosity: float = 0.22
    sw: float = 0.25
    bo: float = 1.3
    mu_o: float = 1.0
    mu_w: float = 0.5
    p_i: float = 4000.0
    p_b: float = 3000.0
    p_a: float = 1000.0
    net_pay_ft: float = 100.0

    def validate(self) -> None:
        if not 0 < self.porosity < 1 or not 0 < self.sw < 1:
            raise RecoveryError("Porosity and water saturation must be fractions between 0 and 1")
        if self.p_a >= self.p_i:
            raise RecoveryError("Abandonment pressure must be below the initial pressure")
        if self.p_a >= self.p_b:
            raise RecoveryError("Abandonment pressure must be below the bubble-point pressure")
        for n in ("bo", "mu_o", "mu_w", "p_i", "p_b", "p_a", "net_pay_ft"):
            if getattr(self, n) <= 0:
                raise RecoveryError(f"{n} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def correlation_rf(name: str, k_md, p: OilBenchmarkInputs, phi=None, sw=None, bo=None) -> np.ndarray:
    """RF for arrays of permeability (and optionally per-trial porosity, Sw, Bo)."""
    p.validate()
    phi = p.porosity if phi is None else phi
    sw = p.sw if sw is None else sw
    bo = p.bo if bo is None else bo
    if name == "api_water":
        return api_water_drive(phi, sw, bo, k_md, p.mu_w, p.mu_o, p.p_i, p.p_a)
    if name == "api_solution_gas":
        return api_solution_gas(phi, sw, bo, k_md, p.mu_o, p.p_b, p.p_a)
    if name == "guthrie_greenberger":
        return guthrie_greenberger(k_md, sw, p.mu_o, phi, p.net_pay_ft)
    raise RecoveryError(f"Unknown correlation '{name}'")


def curves(p: OilBenchmarkInputs, k_min: float = 0.1, k_max: float = 10000.0, n: int = 200,
           names: list[str] | None = None) -> pd.DataFrame:
    if not 0 < k_min < k_max:
        raise RecoveryError("Permeability range must satisfy 0 < min < max")
    k = np.logspace(np.log10(k_min), np.log10(k_max), n)
    df = pd.DataFrame({"Permeability [mD]": k})
    for nm in names or list(CORRELATIONS):
        df[CORRELATIONS[nm]] = correlation_rf(nm, k, p)
    return df


def formula(name: str, p: OilBenchmarkInputs) -> str:
    """Segment formula for rf_oil using the segment's porosity, hc_saturation, bo and permeability [mD]."""
    p.validate()
    if name == "api_water":
        return (f"0.54898 * (porosity * hc_saturation / bo)**0.0422 * (permeability / 1000 * {p.mu_w:g} / {p.mu_o:g})"
                f"**0.0770 * (1 - hc_saturation)**(-0.1903) * ({p.p_i:g} / {p.p_a:g})**(-0.2159)")
    if name == "api_solution_gas":
        return (f"0.41815 * (porosity * hc_saturation / bo)**0.1611 * (permeability / 1000 / {p.mu_o:g})**0.0979"
                f" * (1 - hc_saturation)**0.3722 * ({p.p_b:g} / {p.p_a:g})**0.1741")
    if name == "guthrie_greenberger":
        return (f"0.2719 * log10(permeability) + 0.25569 * (1 - hc_saturation) - 0.1355 * log10({p.mu_o:g})"
                f" - 1.538 * porosity - 0.00035 * {p.net_pay_ft:g} + 0.11403")
    raise RecoveryError(f"Unknown correlation '{name}'")


# ---- analogue tables ----------------------------------------------------------------------

ANALOG_TEMPLATE = "name,permeability_md,rf,drive,fluid\nAnalogue A,250,0.42,water drive,oil\n"


def read_analogs(data: str | bytes) -> pd.DataFrame:
    """CSV with at least permeability (mD) and recovery factor (fraction or %) columns."""
    import io

    text = data.decode("utf-8", "replace") if isinstance(data, bytes) else str(data)
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001 - pandas raises many types
        raise RecoveryError(f"Could not read the analogue file: {exc}") from exc
    cols = {c.lower().strip(): c for c in df.columns}
    kcol = next((cols[c] for c in cols if c.startswith(("perm", "k_", "k ")) or c in ("k", "kh")), None)
    rcol = next((cols[c] for c in cols if c.startswith(("rf", "recovery"))), None)
    if kcol is None or rcol is None:
        raise RecoveryError("Analogue file needs a permeability column (e.g. 'permeability_md') and a recovery "
                            "factor column (e.g. 'rf')")
    out = pd.DataFrame({"Permeability [mD]": pd.to_numeric(df[kcol], errors="coerce"),
                        "RF": pd.to_numeric(df[rcol], errors="coerce")})
    if out["RF"].max() > 1.5:
        out["RF"] = out["RF"] / 100.0
    for key in ("name", "drive", "fluid"):
        c = cols.get(key)
        out[key.capitalize()] = df[c].astype(str) if c else ""
    out = out.dropna(subset=["Permeability [mD]", "RF"])
    out = out[(out["Permeability [mD]"] > 0) & (out["RF"] >= 0) & (out["RF"] <= 1)]
    if out.empty:
        raise RecoveryError("No valid analogue rows (permeability > 0 and RF between 0 and 1)")
    return out.reset_index(drop=True)


def compare_trials(rf_input: np.ndarray, k: np.ndarray, p: OilBenchmarkInputs, names: list[str],
                   phi=None, sw=None, bo=None) -> pd.DataFrame:
    """P90/P50/P10 of the input RF next to each correlation evaluated trial by trial."""
    def pct(a):
        q = np.quantile(a, [0.1, 0.5, 0.9])
        return {"P90": q[0], "P50": q[1], "P10": q[2], "Mean": float(np.mean(a))}

    rows = [{"Source": "Recovery factor input", **pct(rf_input)}]
    for nm in names:
        rows.append({"Source": CORRELATIONS[nm], **pct(correlation_rf(nm, k, p, phi, sw, bo))})
    return pd.DataFrame(rows)
