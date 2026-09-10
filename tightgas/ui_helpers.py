"""
Shared Streamlit widget helpers used by both pages.

Two concerns are handled here, both because of the same underlying
Streamlit rule: a widget created with BOTH a `value=` argument AND a
`key=` whose session_state entry was set programmatically earlier in the
same script run raises StreamlitAPIException. The fix in both cases is
the same pattern: seed a sensible default into session_state only if it
isn't already there, then create the widget from the key alone.

1. Plain widgets (`num_in`, `select_in`, `text_in`) - used wherever a
   value needs to be programmatically overwritten later (e.g. the
   multi-well page's schedule-suggestion tool pushing start days into the
   per-well widgets).
2. Unit-aware widgets (`unit_num_in`) - the underlying value is always
   canonical field units (what the engine consumes); the widget itself
   shows/accepts whatever the currently-selected unit system expects.
   Switching units reconverts the last FIELD-unit value (including a
   user's own edits) into the new system's display units, rather than
   silently resetting to a default.
"""

from __future__ import annotations

import streamlit as st

from .units import to_display, to_field, to_display_delta, to_field_delta, unit_label, FIELD, METRIC


def init_unit_system(location=None) -> tuple[str, bool]:
    """Render the Field/Metric selector (in the sidebar by default) and
    return (system, changed_this_run). `changed_this_run` is used to force
    every unit-aware widget to reconvert its display value this render.
    """
    host = location if location is not None else st.sidebar
    if "unit_system" not in st.session_state:
        st.session_state["unit_system"] = FIELD
    system = host.radio("Units", [FIELD, METRIC], key="unit_system", horizontal=True)
    changed = st.session_state.get("_prev_unit_system") != system
    st.session_state["_prev_unit_system"] = system
    return system, changed


def num_in(container, label, key, default, **kwargs):
    if key not in st.session_state:
        st.session_state[key] = default
    return container.number_input(label, key=key, **kwargs)


def select_in(container, label, options, key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return container.selectbox(label, options, key=key)


def text_in(container, label, key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return container.text_input(label, key=key)


def unit_num_in(container, label, key, field_default, qty, system, changed,
                  min_field=None, max_field=None, step_field=None, fmt=None, help=None):
    """A number_input whose canonical value is always in field units.

    `field_default` seeds the FIRST-ever render. On every subsequent
    render, if the unit system just changed (`changed=True`) or the
    display widget doesn't exist yet, the display widget is reseeded from
    the last known field-unit value (a prior user edit, if any) converted
    into the new system - so customizations survive a unit switch.
    Returns the value in FIELD units, ready to hand straight to the
    engine (WellSpec, SimulationInputs, FieldInputs, ...).
    """
    field_key = f"{key}__field"
    disp_key = f"{key}__disp"

    if field_key not in st.session_state:
        st.session_state[field_key] = field_default
    if changed or disp_key not in st.session_state:
        st.session_state[disp_key] = to_display(st.session_state[field_key], qty, system)

    kwargs = {}
    if min_field is not None:
        kwargs["min_value"] = to_display(min_field, qty, system)
    if max_field is not None:
        kwargs["max_value"] = to_display(max_field, qty, system)
    if step_field is not None:
        kwargs["step"] = to_display_delta(step_field, qty, system)
    if fmt:
        kwargs["format"] = fmt
    if help:
        kwargs["help"] = help

    full_label = f"{label} ({unit_label(qty, system)})" if unit_label(qty, system) else label
    val_disp = container.number_input(full_label, key=disp_key, **kwargs)
    val_field = to_field(val_disp, qty, system)
    st.session_state[field_key] = val_field
    return val_field
