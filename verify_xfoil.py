"""Post-optimization performance check: re-analyzes an optimized propeller
using real XFoil (an external viscous panel-method solver) instead of
NeuralFoil, station by station, and prints a side-by-side comparison
against the NeuralFoil-based BEM result the optimizer actually used.

NeuralFoil is a fast, differentiable surrogate, which is why the optimizer
can run gradient-based optimization through it. XFoil is slower and not
differentiable, so it isn't used inside the optimizer itself -- this
script is a sanity check on how closely the optimized design's NeuralFoil
predictions track a real solver, run separately, after the fact.

Requires the actual XFoil executable (https://web.mit.edu/drela/Public/web/xfoil/),
which is not bundled with AeroSandbox -- point `xfoil_command` (or
XFOIL_COMMAND below) at your xfoil.exe.

Standalone usage:
    python verify_xfoil.py [path/to/optimized_prop.csv]

From main.py:
    set RUN_XFOIL_CHECK = True and XFOIL_COMMAND to your xfoil.exe path in
    the USER CONFIGURATION block -- main.py will run this automatically
    right after exporting optimized_prop.csv.
"""
import sys

import aerosandbox as asb
import aerosandbox.numpy as np

from geometry import load_prop_csv
from BEM import BEMAnalysis

DEFAULT_CSV = "optimized_prop.csv"
DEFAULT_XFOIL_COMMAND = r"C:\Users\adamf\Downloads\XFOIL6.99\xfoil.exe"

MAX_ITERS = 25     # classical BEM fixed-point iterations per station
TOL = 1e-4          # convergence tolerance on a/a' between iterations
RELAXATION = 0.5    # under-relaxation factor on the a/a' update, for stability


def _prandtl_loss(B, R, r_hub, r, phi):
    """Same combined Prandtl tip/hub loss factor as BEM.py's induction
    solver -- see BEM.py for the derivation/rationale.
    """
    f_tip = max((B / 2) * (R - r) / (r * np.sin(phi)), 1e-6)
    F_tip = (2 / np.pi) * np.arccos(np.exp(-f_tip))
    f_hub = max((B / 2) * (r - r_hub) / (r * np.sin(phi)), 1e-6)
    F_hub = (2 / np.pi) * np.arccos(np.exp(-f_hub))
    return F_tip * F_hub


def _solve_station_with_xfoil(airfoil, xfoil_command, B, R, r_hub, r, c, beta, Omega, V,
                               rho, mu, a0, ap0):
    """Classical fixed-point BEM iteration at one station, using real XFoil
    (instead of NeuralFoil) for CL/CD at each iterate -- unlike BEM.py's
    CasADi rootfinder, this doesn't need to be differentiable, since it
    only ever runs on concrete, already-optimized numbers.

    Returns a dict of converged station results, or None if XFoil never
    converged at this station (either it couldn't find a solution at some
    iterate's (alpha, Re), or the fixed-point iteration itself didn't
    settle within MAX_ITERS).
    """
    a, ap = a0, ap0
    phi = alpha = W = None

    for _ in range(MAX_ITERS):
        Vax = V * (1 + a)
        Vtan = Omega * r * (1 - ap)
        phi = np.arctan2(Vax, Vtan)
        W = np.sqrt(Vax ** 2 + Vtan ** 2)
        alpha = beta - phi
        Re = rho * W * c / mu
        mach = W / 340

        xf = asb.XFoil(airfoil=airfoil, Re=float(Re), mach=float(mach),
                        xfoil_command=xfoil_command, verbose=False)
        result = xf.alpha(float(np.degrees(alpha)))

        if len(result.get("CL", [])) == 0:
            return None  # XFoil didn't converge at this (alpha, Re)

        Cl = float(result["CL"][0])
        Cd = float(result["CD"][0])

        F = _prandtl_loss(B, R, r_hub, r, phi)
        sigma = B * c / (2 * np.pi * r)
        Cn = Cl * np.cos(phi) - Cd * np.sin(phi)
        Ct = Cl * np.sin(phi) + Cd * np.cos(phi)

        # Direct algebraic solution of each residual for a/a' individually
        # -- with Cl/Cd/F fixed at this iterate's values, the two residuals
        # decouple and each has a closed-form solution.
        denom_a = 4 * F * np.sin(phi) ** 2 - sigma * Cn
        denom_ap = 4 * F * np.sin(phi) * np.cos(phi) + sigma * Ct
        if abs(denom_a) < 1e-9 or abs(denom_ap) < 1e-9:
            return None  # degenerate station (near-zero denominator)

        a_new = sigma * Cn / denom_a
        ap_new = sigma * Ct / denom_ap

        if abs(a_new - a) < TOL and abs(ap_new - ap) < TOL:
            a, ap = a_new, ap_new
            break

        a += RELAXATION * (a_new - a)
        ap += RELAXATION * (ap_new - ap)
    else:
        return None  # never converged within MAX_ITERS

    q = 0.5 * rho * W ** 2
    Lift = q * c * Cl
    Drag = q * c * Cd
    Fn = Lift * np.cos(phi) - Drag * np.sin(phi)
    Ft = Lift * np.sin(phi) + Drag * np.cos(phi)

    return {
        "a": a, "ap": ap,
        "alpha_deg": float(np.degrees(alpha)),
        "Re": Re, "Cl": Cl, "Cd": Cd,
        "Fn": Fn, "Ft": Ft,
    }


def run_xfoil_check(csv_filepath=DEFAULT_CSV, xfoil_command=DEFAULT_XFOIL_COMMAND):
    prop, rpm, velocity, rho, mu = load_prop_csv(csv_filepath)
    B, R, r_hub = prop.n_blades, prop.radius, prop.hub_radius
    Omega = rpm * 2 * np.pi / 60

    # NeuralFoil-based result, for reference -- exactly what the optimizer
    # actually saw, recomputed here so it's a fair apples-to-apples
    # comparison against the same loaded station grid/operating point.
    nf_results = BEMAnalysis(propeller=prop, rpm=rpm, velocity=velocity, rho=rho, mu=mu).run()

    print(f"Verifying {csv_filepath} against real XFoil "
          f"({prop.n_stations} stations, xfoil_command={xfoil_command!r})")
    print("This can take a while -- XFoil runs as a subprocess, several times per station.\n")

    xfoil_sections = []
    thrust = 0.0
    torque = 0.0

    for i in range(prop.n_stations):
        r = float(prop.r[i])
        c = float(prop.chord[i])
        dr_i = float(prop.dr[i])
        beta = np.radians(float(prop.twist[i]))
        nf_sec = nf_results["sections"][i]

        sec = _solve_station_with_xfoil(
            prop.airfoil, xfoil_command, B, R, r_hub, r, c, beta, Omega, velocity, rho, mu,
            a0=float(nf_sec["a"]), ap0=float(nf_sec["ap"]),
        )

        if sec is None:
            print(f"  station {i:2d} (r={r:.3f} m): XFoil did not converge -- skipped")
            xfoil_sections.append(None)
            continue

        dT = sec["Fn"] * B * dr_i
        dQ = sec["Ft"] * r * B * dr_i
        thrust += dT
        torque += dQ
        xfoil_sections.append(sec)

    power = torque * Omega
    efficiency = thrust * velocity / power if power > 1e-6 else 0.0

    print()
    print(f"{'':>12} {'NeuralFoil (optimized)':>24} {'XFoil (verified)':>20}")
    print(f"{'Thrust':>12} {float(nf_results['thrust']):>21.2f} N {thrust:>17.2f} N")
    print(f"{'Torque':>12} {float(nf_results['torque']):>21.2f} Nm{torque:>17.2f} Nm")
    print(f"{'Power':>12} {float(nf_results['power']):>21.2f} W {power:>17.2f} W")
    print(f"{'Efficiency':>12} {float(nf_results['efficiency']):>23.4f} {efficiency:>19.4f}")
    print()
    print(f"{'r (m)':>8} {'alpha_nf':>10} {'alpha_xf':>10} {'Cl_nf':>8} {'Cl_xf':>8} {'Cd_nf':>8} {'Cd_xf':>8}")
    for i in range(prop.n_stations):
        nf_sec = nf_results["sections"][i]
        xf_sec = xfoil_sections[i]
        if xf_sec is None:
            print(f"{prop.r[i]:8.3f} {float(nf_sec['alpha_deg']):10.2f} {'--':>10} "
                  f"{float(nf_sec['Cl']):8.3f} {'--':>8} {float(nf_sec['Cd']):8.4f} {'--':>8}")
        else:
            print(f"{prop.r[i]:8.3f} {float(nf_sec['alpha_deg']):10.2f} {xf_sec['alpha_deg']:10.2f} "
                  f"{float(nf_sec['Cl']):8.3f} {xf_sec['Cl']:8.3f} {float(nf_sec['Cd']):8.4f} {xf_sec['Cd']:8.4f}")

    return {
        "thrust": thrust, "torque": torque, "power": power, "efficiency": efficiency,
        "sections": xfoil_sections,
    }


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    run_xfoil_check(csv_path, xfoil_command=DEFAULT_XFOIL_COMMAND)
