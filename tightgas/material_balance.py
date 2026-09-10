"""
Volumetric gas-in-place and p/z material balance for a tank-type
(single-layer, no aquifer) tight-gas reservoir, with an optional
formation-compaction correction that is important in many
overpressured / stress-sensitive tight-gas plays.

Standard p/z equation (King, 1993; Ahmed, "Reservoir Engineering
Handbook"):

    p/z = (pi/zi) * (1 - Gp/G)                          [volumetric, no water influx]

Generalized (compaction-corrected) form used here, following the
common engineering extension that lumps rock + connate-water
expansion / pore-volume compaction into an apparent-compressibility
term cf (1/psi):

    (p/z) * (1 + cf*(pi - p)) = (pi/zi) * (1 - Gp/G)

Setting cf = 0 recovers the classical volumetric tank model. cf > 0
produces the concave-down p/z trend commonly observed in abnormally
pressured tight-gas reservoirs (apparent OGIP overestimated if a
straight line is forced through the data).
"""

from __future__ import annotations

from dataclasses import dataclass

from .pvt import GasPVT


def ogip_volumetric(area_acres: float, h_ft: float, phi: float, sw: float,
                     bgi_rcf_per_scf: float) -> float:
    """Original gas in place, scf, from static (volumetric) parameters.

    G = 43,560 * A[ac] * h[ft] * phi * (1-Sw) / Bgi
    (43,560 ft^3 per acre-ft; Bgi in rcf/scf so the ft^3 units cancel.)
    """
    pore_volume_ft3 = 43_560.0 * area_acres * h_ft * phi * (1.0 - sw)
    return pore_volume_ft3 / bgi_rcf_per_scf


@dataclass
class MaterialBalanceTank:
    pvt: GasPVT
    G_scf: float          # original gas in place
    pi: float              # initial reservoir pressure, psia
    cf: float = 0.0        # apparent compaction/water-drive correction, 1/psi

    def __post_init__(self):
        self.zi = float(self.pvt.z(self.pi))
        self.poz_i = self.pi / self.zi

    def p_from_gp(self, gp_scf: float, p_lo: float = 25.0) -> float:
        """Invert the (compaction-corrected) p/z material balance for average
        reservoir pressure p, given cumulative production Gp (scf).
        Robust bisection (z(p) and cf term are both well-behaved/monotonic
        over the tight-gas pressure range).
        """
        gp_scf = min(gp_scf, 0.999999 * self.G_scf)
        target_ratio = 1.0 - gp_scf / self.G_scf  # RHS / (pi/zi)

        def f(p):
            z = float(self.pvt.z(p))
            lhs = (p / z) * (1.0 + self.cf * (self.pi - p)) / self.poz_i
            return lhs - target_ratio

        lo, hi = p_lo, self.pi
        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi > 0:
            # fall back to a fine scan if bisection brackets fail
            import numpy as np
            grid = np.linspace(lo, hi, 400)
            vals = np.array([f(p) for p in grid])
            idx = int(np.argmin(np.abs(vals)))
            return float(grid[idx])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if f_lo * f_mid <= 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
            if abs(hi - lo) < 1e-3:
                break
        return 0.5 * (lo + hi)

    def gp_from_p(self, p: float) -> float:
        """Cumulative production corresponding to average reservoir pressure p."""
        z = float(self.pvt.z(p))
        ratio = (p / z) * (1.0 + self.cf * (self.pi - p)) / self.poz_i
        ratio = min(max(ratio, 0.0), 1.0)
        return (1.0 - ratio) * self.G_scf

    def poz(self, p: float) -> float:
        return p / float(self.pvt.z(p))
