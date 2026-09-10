"""
Field <-> Metric unit conversion registry.

All internal engineering calculations throughout `tightgas/` are performed
in U.S. field units (psia, degR/degF, ft, md, Mscf/d, bbl, acres) - the
correlations in pvt.py (Dranchuk-Abu-Kassem, Lee-Gonzalez-Eakin) and the
flow equations in flow_models.py are field-unit correlations by
construction, and are NOT touched by the unit system the user selects.
This module only converts values at the presentation layer: what a widget
displays/accepts, and what a plot/table/export shows.

Conventions chosen for "Metric" (documented so a user comparing against
another metric tool knows exactly what's assumed):

    pressure        psia            <-> kPa(a)
    temperature     degF            <-> degC
    length          ft              <-> m
    area            acres           <-> hectares
    permeability    md              <-> md   (unchanged - universal unit)
    gas rate        Mscf/d          <-> 10^3 m3/d   ("e3m3/d")
    gas rate (MM)   MMscf/d         <-> 10^3 m3/d   ("e3m3/d")
    gas volume      Bcf             <-> 10^6 m3     ("Mm3" / "e6m3")
    liquid rate     bbl/d (STB/d)   <-> m3/d
    liquid volume   Mstb (10^3 bbl) <-> 10^3 m3      ("e3m3")
    CGR / yield     STB/MMscf       <-> m3/(10^3 m3 gas)
    dimensionless   -               <-> -            (skin, b, FcD, RF%, n_frac, ...)
"""

from __future__ import annotations

# Multiplicative field->metric factors (value_metric = value_field * factor).
# Temperature is handled separately (affine, not linear).
_FACTORS = {
    "pressure": 6.894757,          # psia -> kPa
    "inv_pressure": 1.0 / 6.894757, # 1/psi -> 1/kPa
    "length": 0.3048,               # ft -> m
    "area": 0.404686,               # acres -> hectares
    "perm": 1.0,                    # md -> md (no-op)
    "rate_gas_mscfd": 0.0283168,    # Mscf/d -> 10^3 m3/d
    "rate_gas_mmscfd": 28.3168,     # MMscf/d -> 10^3 m3/d
    "volume_gas_bcf": 28.3168,      # Bcf -> 10^6 m3
    "rate_liquid_bbld": 0.158987,   # bbl/d -> m3/d
    "volume_liquid_mstb": 0.158987, # Mstb (10^3 bbl) -> 10^3 m3
    "cgr": 0.0056146,               # STB/MMscf -> m3/(10^3 m3 gas)
    "dimensionless": 1.0,
}

_LABELS = {
    "pressure": ("psia", "kPa"),
    "inv_pressure": ("1/psi", "1/kPa"),
    "temperature": ("degF", "degC"),
    "length": ("ft", "m"),
    "area": ("acres", "ha"),
    "perm": ("md", "md"),
    "rate_gas_mscfd": ("Mscf/d", "10^3 m3/d"),
    "rate_gas_mmscfd": ("MMscf/d", "10^3 m3/d"),
    "volume_gas_bcf": ("Bcf", "10^6 m3"),
    "rate_liquid_bbld": ("bbl/d", "m3/d"),
    "volume_liquid_mstb": ("Mstb", "10^3 m3"),
    "cgr": ("STB/MMscf", "m3/10^3m3"),
    "dimensionless": ("", ""),
}

FIELD = "Field"
METRIC = "Metric"


def unit_label(qty: str, system: str) -> str:
    field_lbl, metric_lbl = _LABELS[qty]
    return metric_lbl if system == METRIC else field_lbl


def to_display(value_field: float, qty: str, system: str) -> float:
    """Convert a value stored internally in field units to the unit the
    given display system expects."""
    if value_field is None:
        return None
    if system != METRIC:
        return value_field
    if qty == "temperature":
        return (value_field - 32.0) * 5.0 / 9.0
    return value_field * _FACTORS[qty]


def to_field(value_display: float, qty: str, system: str) -> float:
    """Convert a value entered in the display system back to field units
    (the only units the engine ever sees)."""
    if value_display is None:
        return None
    if system != METRIC:
        return value_display
    if qty == "temperature":
        return value_display * 9.0 / 5.0 + 32.0
    return value_display / _FACTORS[qty]


def to_display_delta(delta_field: float, qty: str, system: str) -> float:
    """Convert a step/increment (no affine offset, even for temperature)."""
    if delta_field is None:
        return None
    if system != METRIC:
        return delta_field
    if qty == "temperature":
        return delta_field * 5.0 / 9.0
    return delta_field * _FACTORS[qty]


def to_field_delta(delta_display: float, qty: str, system: str) -> float:
    if delta_display is None:
        return None
    if system != METRIC:
        return delta_display
    if qty == "temperature":
        return delta_display * 9.0 / 5.0
    return delta_display / _FACTORS[qty]
