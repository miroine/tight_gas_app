"""
Shared chart styling for the Streamlit pages, following the project's
data-visualization guidelines: a fixed-order, validated-safe categorical
palette (never cycled/reassigned when a filter changes series count), a
single-hue sequential ramp for magnitude, and a blue<->red diverging pair
for polarity (used by the sensitivity tornado chart).

Colors are the reference validated palette (see the dataviz skill's
palette.md) - worst-case adjacent CVD Delta E 9.1 (light) / 8.4 (dark),
worst-case adjacent normal-vision Delta E 19.6 (light) / 19.3 (dark).
"""

from __future__ import annotations

# Fixed-order categorical palette - assign by position, never re-sort/cycle
# when the number of series (wells, well types) changes.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Single-hue sequential ramp (blue), light -> dark, for magnitude encodings.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

# Diverging pair for polarity (e.g. sensitivity tornado: increase vs decrease
# from a base case), with a neutral gray midpoint.
DIVERGING_POS = "#2a78d6"   # blue = increase
DIVERGING_NEG = "#e34948"   # red = decrease
NEUTRAL_GRAY = "#c3c2b7"

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

MUTED_INK = "#898781"
SECONDARY_INK = "#52514e"
GRIDLINE = "#e1e0d9"


def series_color(i: int) -> str:
    """Fixed-order categorical color for series index i (wraps if >8, but
    per the palette's own guidance a chart with that many series should
    fold extras into 'Other' or facet rather than lean on this wraparound)."""
    return CATEGORICAL[i % len(CATEGORICAL)]


def style_fig(fig, height: int = 420, hovermode: str = "x unified"):
    """Apply consistent, theme-agnostic styling: transparent surfaces (so
    the chart blends into Streamlit's own light/dark container), muted
    gridlines/axes, thin recessive chrome, legend always present for
    multi-series charts.
    """
    fig.update_layout(
        height=height,
        hovermode=hovermode,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=SECONDARY_INK, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                     bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=45, b=10),
        colorway=CATEGORICAL,
    )
    fig.update_xaxes(gridcolor=GRIDLINE, zeroline=False, showline=True,
                       linecolor=GRIDLINE, title_font=dict(color=MUTED_INK))
    fig.update_yaxes(gridcolor=GRIDLINE, zeroline=False, showline=True,
                       linecolor=GRIDLINE, title_font=dict(color=MUTED_INK))
    return fig
