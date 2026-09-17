"""Unit system.

The engine works internally in oilfield units, whatever the display system:

    area              acres
    length / depth    ft
    GRV               acre-ft
    oil / condensate  stb
    gas               scf
    oil equivalent    boe
    Bo                rb/stb
    Bg                rcf/scf            (dimensionless; identical to rm3/Sm3)
    GOR               scf/stb
    CGR               stb/MMscf

All conversion happens at the UI / file boundary through :func:`to_internal`
and :func:`from_internal`. Standard-condition differences between scf and Sm3
(60 degF / 14.696 psia vs 15 degC / 101.325 kPa) are ignored, as is usual in
screening-level volumetrics.
"""
from __future__ import annotations

import numpy as np

# Exact / defined constants
FT_M = 0.3048
ACRE_M2 = 4046.8564224
BBL_M3 = 0.158987294928
FT3_M3 = FT_M ** 3  # 0.028316846592

# Volumetric constants in oilfield units
BBL_PER_ACRE_FT = ACRE_M2 * FT_M / BBL_M3  # 7758.37
FT3_PER_ACRE_FT = 43560.0

# Gas-to-oil-equivalent conventions (scf of gas per boe)
BOE_CONVENTIONS = {
    "NCS (1000 Sm3 gas = 1 Sm3 o.e.)": 1000.0 * (1.0 / FT3_M3) / (1.0 / BBL_M3),  # 5614.6
    "SPE (6 Mscf = 1 boe)": 6000.0,
}
DEFAULT_BOE_CONVENTION = "NCS (1000 Sm3 gas = 1 Sm3 o.e.)"

# factor: multiply a value in `unit` by factor to obtain internal units.
# The first unit of each quantity is the internal unit (factor 1).
_UNITS: dict[str, dict[str, float]] = {
    "area": {"acres": 1.0, "km2": 1.0e6 / ACRE_M2, "ha": 1.0e4 / ACRE_M2, "m2": 1.0 / ACRE_M2, "mi2": 640.0},
    "length": {"ft": 1.0, "m": 1.0 / FT_M},
    "grv": {"acre-ft": 1.0, "1e3 acre-ft": 1.0e3, "m3": 1.0 / (ACRE_M2 * FT_M), "1e6 m3": 1.0e6 / (ACRE_M2 * FT_M),
            "1e9 m3": 1.0e9 / (ACRE_M2 * FT_M), "ft3": 1.0 / 43560.0},
    "oil": {"stb": 1.0, "Mstb": 1.0e3, "MMstb": 1.0e6, "Sm3": 1.0 / BBL_M3, "1e3 Sm3": 1.0e3 / BBL_M3,
            "MSm3": 1.0e6 / BBL_M3},
    "gas": {"scf": 1.0, "Mscf": 1.0e3, "MMscf": 1.0e6, "Bscf": 1.0e9, "Tscf": 1.0e12, "Sm3": 1.0 / FT3_M3,
            "1e3 Sm3": 1.0e3 / FT3_M3, "MSm3": 1.0e6 / FT3_M3, "GSm3": 1.0e9 / FT3_M3},
    "oe": {"boe": 1.0, "Mboe": 1.0e3, "MMboe": 1.0e6, "Sm3 o.e.": 1.0 / BBL_M3, "1e3 Sm3 o.e.": 1.0e3 / BBL_M3,
           "MSm3 o.e.": 1.0e6 / BBL_M3},
    "gor": {"scf/stb": 1.0, "Sm3/Sm3": (1.0 / FT3_M3) / (1.0 / BBL_M3), "Mscf/stb": 1.0e3},
    "cgr": {"stb/MMscf": 1.0, "Sm3/MSm3": (1.0 / BBL_M3) / ((1.0e6 / FT3_M3) / 1.0e6),
            "Sm3/Sm3": (1.0 / BBL_M3) / ((1.0 / FT3_M3) / 1.0e6), "stb/Mscf": 1.0e3},
    "bo": {"rb/stb": 1.0, "Rm3/Sm3": 1.0},
    "bg": {"rcf/scf": 1.0, "Rm3/Sm3": 1.0, "rb/Mscf": (BBL_M3 / FT3_M3) / 1.0e3},
    "eg": {"scf/rcf": 1.0, "Sm3/Rm3": 1.0},
    "fraction": {"fraction": 1.0, "%": 0.01},
    "none": {"-": 1.0},
}

QUANTITY_LABELS = {
    "area": "Area", "length": "Depth & thickness", "grv": "Gross rock volume", "oil": "Oil & condensate volume",
    "gas": "Gas volume", "oe": "Oil equivalent", "gor": "Gas-oil ratio", "cgr": "Condensate-gas ratio",
    "bo": "Oil formation volume factor", "bg": "Gas formation volume factor", "eg": "Gas expansion factor",
    "fraction": "Fractions (N/G, porosity, saturation, RF)",
}

UNIT_SYSTEMS = {
    "Metric (NCS)": {
        "area": "km2", "length": "m", "grv": "1e6 m3", "oil": "MSm3", "gas": "GSm3",
        "oe": "MSm3 o.e.", "gor": "Sm3/Sm3", "cgr": "Sm3/MSm3", "bo": "Rm3/Sm3", "bg": "Rm3/Sm3", "eg": "Sm3/Rm3",
        "fraction": "fraction", "none": "-",
    },
    "Field": {
        "area": "acres", "length": "ft", "grv": "acre-ft", "oil": "MMstb", "gas": "Bscf",
        "oe": "MMboe", "gor": "scf/stb", "cgr": "stb/MMscf", "bo": "rb/stb", "bg": "rcf/scf", "eg": "scf/rcf",
        "fraction": "fraction", "none": "-",
    },
}


def internal_unit(quantity: str) -> str:
    return next(iter(_UNITS[quantity]))


def is_valid_unit(quantity: str, unit: str) -> bool:
    return quantity in _UNITS and unit in _UNITS[quantity]


def resolve_unit(system: str, quantity: str, overrides: dict[str, str] | None = None) -> str:
    """Display unit: explicit override if valid, else the preset system's unit."""
    if overrides and is_valid_unit(quantity, overrides.get(quantity, "")):
        return overrides[quantity]
    return UNIT_SYSTEMS.get(system, UNIT_SYSTEMS["Metric (NCS)"])[quantity]


def units_for(quantity: str) -> list[str]:
    return list(_UNITS[quantity].keys())


def factor(quantity: str, unit: str) -> float:
    try:
        return _UNITS[quantity][unit]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown unit '{unit}' for quantity '{quantity}'") from exc


def to_internal(value, quantity: str, unit: str):
    return np.asarray(value, dtype=float) * factor(quantity, unit) if not np.isscalar(value) \
        else float(value) * factor(quantity, unit)


def from_internal(value, quantity: str, unit: str):
    return np.asarray(value, dtype=float) / factor(quantity, unit) if not np.isscalar(value) \
        else float(value) / factor(quantity, unit)


def display_unit(system: str, quantity: str) -> str:
    return resolve_unit(system, quantity)
