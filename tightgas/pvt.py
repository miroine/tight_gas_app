"""
Real-gas PVT correlations for dry / tight gas systems.

All functions use U.S. field units unless noted:
    pressure      psia
    temperature   °R  (helper converts from °F)
    gas gravity   dimensionless (air = 1.0)

References
----------
Standing, M.B. (1977) "Volumetric and Phase Behavior of Oil Field Hydrocarbon Systems"
    -> pseudo-critical property correlation (sweet gas)
Wichert, E. & Aziz, K. (1972) "Calculate Zs for sour gases", Hydrocarbon Processing
    -> non-hydrocarbon (CO2, H2S) correction to pseudo-criticals
Dranchuk, P.M. & Abu-Kassem, J.H. (1975) "Calculation of Z Factors for Natural
    Gases Using Equations of State", JCPT
    -> Z-factor equation of state fit to the Standing-Katz chart
Lee, A.L., Gonzalez, M.H., Eakin, B.E. (1966) "The Viscosity of Natural Gases",
    JPT -> gas viscosity correlation
Al-Hussainy, R., Ramey, H.J., Crawford, P.B. (1966) "The Flow of Real Gases
    Through Porous Media" -> real-gas pseudo-pressure m(p)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

R_FT3 = 10.732  # psia-ft3/(lbmol-R), gas constant in field units
TSC = 520.0     # °R (60 F)
PSC = 14.696    # psia


def f_to_r(t_f: float) -> float:
    return t_f + 459.67


@dataclass
class GasComposition:
    gamma_g: float = 0.65   # specific gravity, air = 1
    y_co2: float = 0.0      # mole fraction CO2
    y_n2: float = 0.0       # mole fraction N2
    y_h2s: float = 0.0      # mole fraction H2S

    @property
    def molecular_weight(self) -> float:
        return 28.97 * self.gamma_g


def pseudo_criticals_standing(gamma_g: float) -> tuple[float, float]:
    """Standing (1977) sweet-gas pseudo-critical pressure & temperature.

    Returns (Ppc [psia], Tpc [degR])
    """
    ppc = 677.0 + 15.0 * gamma_g - 37.5 * gamma_g ** 2
    tpc = 168.0 + 325.0 * gamma_g - 12.5 * gamma_g ** 2
    return ppc, tpc


def wichert_aziz_correction(ppc: float, tpc: float, y_co2: float, y_h2s: float) -> tuple[float, float]:
    """Correct pseudo-criticals for CO2 / H2S content (sour-gas correction).

    Returns corrected (Ppc', Tpc').
    """
    a = y_co2 + y_h2s
    b = y_h2s
    if a <= 0.0:
        return ppc, tpc
    eps = 120.0 * (a ** 0.9 - a ** 1.6) + 15.0 * (b ** 0.5 - b ** 4.0)
    tpc_corr = tpc - eps
    ppc_corr = ppc * tpc_corr / (tpc + b * (1.0 - b) * eps)
    return ppc_corr, tpc_corr


def pseudo_criticals(comp: GasComposition) -> tuple[float, float]:
    """Full pipeline: Standing correlation + Wichert-Aziz sour correction."""
    ppc, tpc = pseudo_criticals_standing(comp.gamma_g)
    ppc, tpc = wichert_aziz_correction(ppc, tpc, comp.y_co2, comp.y_h2s)
    return ppc, tpc


# Dranchuk & Abu-Kassem (1975) constants
_A = [0.3265, -1.0700, -0.5339, 0.01569, -0.05165, 0.5475,
      -0.7361, 0.1844, 0.1056, 0.6134, 0.7210]


def z_factor_dak(p: float, t: float, ppc: float, tpc: float) -> float:
    """Dranchuk & Abu-Kassem (1975) real-gas Z-factor, solved by Newton-Raphson.

    p, t in psia / degR (absolute). ppc, tpc are pseudo-critical psia / degR.
    Valid ~0.2 <= Ppr <= 30, 1.0 <= Tpr <= 3.0 (typical tight-gas conditions
    fall well inside this range).
    """
    ppr = p / ppc
    tpr = t / tpc
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = _A

    c1 = a1 + a2 / tpr + a3 / tpr ** 3 + a4 / tpr ** 4 + a5 / tpr ** 5
    c2 = a6 + a7 / tpr + a8 / tpr ** 2
    c3 = a9 * (a7 / tpr + a8 / tpr ** 2)

    def residual(z):
        rho_r = 0.27 * ppr / (z * tpr)
        f = (1 + c1 * rho_r + c2 * rho_r ** 2 - c3 * rho_r ** 5
             + a10 * (1 + a11 * rho_r ** 2) * (rho_r ** 2 / tpr ** 3)
             * np.exp(-a11 * rho_r ** 2)) - z
        return f

    z = 1.0
    for _ in range(60):
        h = 1e-6
        f0 = residual(z)
        f1 = residual(z + h)
        deriv = (f1 - f0) / h
        if abs(deriv) < 1e-12:
            break
        z_new = z - f0 / deriv
        z_new = max(z_new, 0.05)
        if abs(z_new - z) < 1e-8:
            z = z_new
            break
        z = z_new
    return float(z)


def gas_viscosity_lgc(p: float, t: float, z: float, mw: float) -> float:
    """Lee-Gonzalez-Eakin (1966) gas viscosity, cp.

    p in psia, t in degR, z dimensionless, mw = apparent molecular weight.
    """
    rho_g = 1.4935e-3 * p * mw / (z * t)  # g/cc
    k = (9.4 + 0.02 * mw) * t ** 1.5 / (209.0 + 19.0 * mw + t)
    x = 3.5 + 986.0 / t + 0.01 * mw
    y = 2.4 - 0.2 * x
    mu = 1e-4 * k * np.exp(x * rho_g ** y)
    return float(mu)


def bg_rcf_per_scf(p: float, t: float, z: float) -> float:
    """Gas formation volume factor, rcf/scf."""
    return 0.02827 * z * t / p


def gas_compressibility(p: float, t: float, ppc: float, tpc: float, z: float | None = None) -> float:
    """cg = 1/p - (1/z) dz/dp, 1/psi, via central finite difference on DAK Z."""
    if z is None:
        z = z_factor_dak(p, t, ppc, tpc)
    dp = max(p * 1e-4, 0.05)
    z_hi = z_factor_dak(p + dp, t, ppc, tpc)
    z_lo = z_factor_dak(p - dp, t, ppc, tpc)
    dzdp = (z_hi - z_lo) / (2 * dp)
    cg = 1.0 / p - dzdp / z
    return float(cg)


class GasPVT:
    """Convenience wrapper: builds Z, mu, Bg, cg, m(p) tables for a given gas
    over a pressure range and provides fast interpolated lookups.
    """

    def __init__(self, comp: GasComposition, temp_f: float,
                 p_min: float = 25.0, p_max: float = 12000.0, n: int = 481):
        self.comp = comp
        self.t = f_to_r(temp_f)
        self.ppc, self.tpc = pseudo_criticals(comp)
        self.mw = comp.molecular_weight

        self.p_grid = np.linspace(p_min, p_max, n)
        z_vals = np.array([z_factor_dak(p, self.t, self.ppc, self.tpc) for p in self.p_grid])
        mu_vals = np.array([gas_viscosity_lgc(p, self.t, z, self.mw)
                             for p, z in zip(self.p_grid, z_vals)])
        cg_vals = np.array([gas_compressibility(p, self.t, self.ppc, self.tpc, z)
                             for p, z in zip(self.p_grid, z_vals)])
        bg_vals = np.array([bg_rcf_per_scf(p, self.t, z) for p, z in zip(self.p_grid, z_vals)])

        self.z_grid = z_vals
        self.mu_grid = mu_vals
        self.cg_grid = cg_vals
        self.bg_grid = bg_vals

        # real-gas pseudo-pressure m(p) = 2 * integral(p/(mu z)) dp, from p_grid[0]
        integrand = 2.0 * self.p_grid / (self.mu_grid * self.z_grid)
        self.mp_grid = np.concatenate(([0.0], np.cumsum(
            (integrand[1:] + integrand[:-1]) / 2.0 * np.diff(self.p_grid))))

    def z(self, p) -> np.ndarray:
        return np.interp(p, self.p_grid, self.z_grid)

    def mu(self, p) -> np.ndarray:
        return np.interp(p, self.p_grid, self.mu_grid)

    def cg(self, p) -> np.ndarray:
        return np.interp(p, self.p_grid, self.cg_grid)

    def bg(self, p) -> np.ndarray:
        return np.interp(p, self.p_grid, self.bg_grid)

    def m(self, p) -> np.ndarray:
        """Real-gas pseudo-pressure, psi^2/cp, referenced to p_grid[0] (~0 psia)."""
        return np.interp(p, self.p_grid, self.mp_grid)

    def dm(self, p1, p2) -> np.ndarray:
        """m(p1) - m(p2)."""
        return self.m(p1) - self.m(p2)
