"""
Gas-well deliverability models for three completion styles:
    - Vertical well (radial flow, optionally fractured via skin)
    - Horizontal well with multi-stage transverse hydraulic fractures (MFHW)
    - Fishbone / multilateral well

Methodology
-----------
Every well type is handled with the same two-regime construction, which is
standard practice in rate-transient analysis (it is, e.g., exactly how
Fetkovich's original type curves are built by splicing a transient stem to
a depletion/PSS stem):

  1. TRANSIENT stem - early/mid time, infinite (or not-yet-felt-boundary)
     acting flow. Governs the *shape* of the decline while the pressure
     transient is still expanding through the matrix / into the fractures.
  2. BOUNDARY-DOMINATED (pseudo-steady-state, "PSS") stem - once the
     transient has reached the edge of the well's drainage volume, the
     rate is controlled by depletion of that volume, i.e. directly by the
     p/z material balance. This stem controls ultimate recovery and is on
     solid physical footing (classical Darcy PSS radial-flow equation).

  At every time step the two are combined as  q = min(q_transient, q_pss)
  evaluated at the *current* average reservoir pressure. This reproduces
  the classic hyperbola-like transition from transient to depletion decline
  and is numerically robust (monotonic, no discontinuities).

All "1422" / "1637" constants below are the standard U.S. field-unit real
gas pseudo-pressure flow constants used throughout gas well testing
(Lee & Wattenbarger, "Gas Reservoir Engineering", SPE Textbook Series):

    q_g [Mscf/d] = k[md]*h[ft]*(m(p1)-m(p2))[psi^2/cp] / (1422 * T[degR] * [dimensionless group])

Multi-fractured horizontal wells and fishbones are NOT solved with a full
trilinear/numerical model here. Instead:
  * Transient stem: analytical constant-pwf linear-flow solution into the
    fracture (or lateral) faces (Wattenbarger square-root-of-time flow
    regime - the dominant regime for most of a tight-gas MFHW's early
    life). A user-adjustable "linear-flow calibration" multiplier is
    exposed because in practice xf and k are rarely both known
    independently - this parameter is exactly what gets history-matched
    against real rate data in commercial RTA software.
  * PSS stem: the fractured/lateral system is converted to an *effective
    wellbore radius* r'w (Cinco-Ley type asymptotic behavior: r'w -> xf/2
    for high fracture conductivity) and then the ordinary vertical-well PSS
    radial-flow equation is reused with that r'w and the well's drainage
    radius. This is a widely used engineering simplification for
    long-term (depletion-dominated) forecasting of fractured horizontals.

Fishbone wells reuse the MFHW machinery: each branch is treated as one
low-conductivity "pseudo-fracture" of half-length = branch length / 2.
This is a simplified, explicitly-flagged approximation - rigorous
analytical fishbone RTA is not standardized in the literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .pvt import GasPVT


class Regime(str, Enum):
    TRANSIENT = "transient"
    BOUNDARY_DOMINATED = "boundary-dominated"


@dataclass
class ReservoirRock:
    k_md: float          # initial matrix permeability, md
    phi: float            # porosity, fraction
    h_ft: float            # net pay thickness, ft
    stress_gamma: float = 0.0   # permeability modulus, 1/psi (stress-dependent k)

    def k_at(self, pi: float, p: float) -> float:
        """Stress-dependent permeability: k = ki * exp(-gamma*(pi-p))."""
        return self.k_md * np.exp(-self.stress_gamma * max(pi - p, 0.0))


@dataclass
class VerticalWell:
    rw_ft: float
    skin: float
    drainage_area_acres: float

    label: str = "Vertical well"

    def re_ft(self) -> float:
        return np.sqrt(43_560.0 * self.drainage_area_acres / np.pi)

    def rate(self, pvt: GasPVT, rock: ReservoirRock, pi: float, p_avg: float, pwf: float,
              t_days: float, mu_i: float, cti: float) -> tuple[float, str]:
        """Hard regime switch at the classical tDA = 0.0002637*k*t/(phi*mu*ct*A) = 0.1
        pseudo-steady-state criterion (Lee, "Well Testing", for a well centered
        in a circular/square drainage area). Before that time the transient
        radial-flow equation (driven by the *initial* pressure, since
        depletion is negligible while the transient is still infinite-acting)
        applies; after it, the boundary-dominated PSS equation (driven by the
        *current* material-balance average pressure) applies. A hard switch
        is used rather than min()/max() of the two formulas because the PSS
        equation is simply not valid before the boundary has been felt - the
        two curves are not comparable point-by-point before that time.
        """
        k = rock.k_at(pi, p_avg)
        h = rock.h_ft
        T = pvt.t
        A_ft2 = 43_560.0 * self.drainage_area_acres

        t_hr = max(t_days, 1e-4) * 24.0
        tDA = 0.0002637 * k * t_hr / (rock.phi * mu_i * cti * A_ft2)

        if tDA < 0.1:
            log_arg = max(k * t_hr / (1688.0 * rock.phi * mu_i * cti * self.rw_ft ** 2), 1.0)
            dm_trans = pvt.dm(pi, pwf)
            denom = 1637.0 * T * (np.log10(log_arg) + 0.87 * self.skin)
            q = max(k * h * dm_trans / denom, 0.0) if denom > 0 else 0.0
            return q, Regime.TRANSIENT

        re = self.re_ft()
        dm_pss = pvt.dm(p_avg, pwf)
        q = k * h * dm_pss / (1422.0 * T * (np.log(re / self.rw_ft) - 0.75 + self.skin))
        return max(q, 0.0), Regime.BOUNDARY_DOMINATED


def effective_wellbore_radius(xf_ft: float, fcd: float) -> float:
    """Approximate effective (equivalent) wellbore radius of a single
    finite-conductivity vertical fracture, asymptoting to the classical
    Prats (1961) infinite-conductivity result r'w = xf/2.

    r'w/xf = 0.5 * FcD / (FcD + 2)

    This is a smooth engineering fit reproducing the correct end-point
    (r'w -> xf/2 as FcD -> inf) and qualitatively correct low-FcD
    behavior; it is NOT a substitute for the tabulated Cinco-Ley &
    Samaniego (1981) solution and is flagged as an approximation.
    """
    fcd = max(fcd, 1e-6)
    return 0.5 * xf_ft * fcd / (fcd + 2.0)


@dataclass
class MultiFracHorizontalWell:
    lateral_length_ft: float
    n_frac: int
    xf_ft: float                 # fracture half-length, ft
    fcd: float                   # dimensionless fracture conductivity
    drainage_area_acres: float   # total area attributed to the well (spacing x length)
    linear_calib: float = 1.0    # calibration multiplier on the linear-flow constant
    transition_tD: float = 0.25  # tDxf at which transient linear flow hands off to PSS

    label: str = "Horizontal multi-frac well"

    def re_ft(self) -> float:
        return np.sqrt(43_560.0 * self.drainage_area_acres / np.pi)

    def rate(self, pvt: GasPVT, rock: ReservoirRock, pi: float, p_avg: float, pwf: float,
              t_days: float, mu_i: float, cti: float) -> tuple[float, str]:
        """Hard regime switch (see VerticalWell.rate docstring for rationale):
        transient formation-linear-flow (driven by initial pressure) while
        tDxf < transition_tD, then boundary-dominated PSS (effective-
        wellbore-radius analogy, driven by the current average pressure)
        afterwards.
        """
        k = rock.k_at(pi, p_avg)
        h = rock.h_ft
        T = pvt.t

        t_hr = max(t_days, 1e-4) * 24.0
        tDxf = 0.0002637 * k * t_hr / (rock.phi * mu_i * cti * self.xf_ft ** 2)
        tDxf = max(tDxf, 1e-8)

        if tDxf < self.transition_tD:
            qD = self.linear_calib * (1.0 / np.sqrt(np.pi * tDxf))
            dm_trans = pvt.dm(pi, pwf)
            q_per_frac = qD * k * h * dm_trans / (1422.0 * T)
            q = max(self.n_frac * q_per_frac, 0.0)
            return q, Regime.TRANSIENT

        rwe = effective_wellbore_radius(self.xf_ft, self.fcd)
        re = self.re_ft()
        dm_pss = pvt.dm(p_avg, pwf)
        q = k * h * dm_pss / (1422.0 * T * (np.log(re / rwe) - 0.75))
        return max(q, 0.0), Regime.BOUNDARY_DOMINATED


@dataclass
class FishboneWell:
    main_bore_length_ft: float
    n_branches: int
    branch_length_ft: float
    drainage_area_acres: float
    linear_calib: float = 0.5    # branches are typically unstimulated -> lower default
    transition_tD: float = 0.25

    label: str = "Fishbone / multilateral well"

    def _as_mfhw(self) -> MultiFracHorizontalWell:
        # Each branch approximated as one low-conductivity "pseudo-fracture"
        # of half-length = branch_length/2, with a moderate default FcD
        # representative of an open-hole/unstimulated lateral.
        return MultiFracHorizontalWell(
            lateral_length_ft=self.main_bore_length_ft,
            n_frac=max(self.n_branches, 1),
            xf_ft=max(self.branch_length_ft / 2.0, 1.0),
            fcd=5.0,
            drainage_area_acres=self.drainage_area_acres,
            linear_calib=self.linear_calib,
            transition_tD=self.transition_tD,
        )

    def re_ft(self) -> float:
        return self._as_mfhw().re_ft()

    def rate(self, pvt: GasPVT, rock: ReservoirRock, pi: float, p_avg: float, pwf: float,
              t_days: float, mu_i: float, cti: float) -> tuple[float, str]:
        return self._as_mfhw().rate(pvt, rock, pi, p_avg, pwf, t_days, mu_i, cti)


WellType = VerticalWell | MultiFracHorizontalWell | FishboneWell
