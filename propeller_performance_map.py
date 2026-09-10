"""Off-design performance map for the already-optimized propeller: thrust,
power, and efficiency across a range of flight speeds, at several RPM
lines spanning plausible pilot-cadence variation around the optimized
cruise design point. This is the propeller's own performance envelope,
not full aircraft climb/descent/takeoff performance (that would need
aircraft mass, a drag polar, and a pilot power-output model, none of
which exist in this project).

Uses NeuralFoil (not XFoil) since this sweeps many operating points and
needs to be fast -- the same aero model the optimizer itself trusted.

Caveats:
- Low-speed/static thrust: BEM.py's velocity triangle (Vax = V*(1+a))
  degenerates as V -> 0, so this sweep stays at a moderate fraction of
  cruise speed and does not predict true static (V=0) thrust.
- The optimizer only enforced its alpha/analysis_confidence bounds at the
  single cruise design point. Away from that point, some stations may sit
  outside NeuralFoil's trusted region -- flagged on the plot as a shaded
  band, not silently trusted.

Usage: python propeller_performance_map.py [path/to/optimized_prop.csv]
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import aerosandbox.numpy as np
import matplotlib.pyplot as plt

from geometry import load_prop_csv
from BEM import BEMAnalysis

OUTPUT_DIR = "outputs"
DEFAULT_CSV = os.path.join(OUTPUT_DIR, "optimized_prop.csv")
DEFAULT_PLOT_PATH = os.path.join(OUTPUT_DIR, "propeller_performance_map.png")

# Same thresholds main.py's optimizer enforces at the design point -- keep
# these in sync with main.py's USER CONFIGURATION if you change them there.
ALPHA_MIN_DEG, ALPHA_MAX_DEG = -4, 10
MIN_ANALYSIS_CONFIDENCE = 0.9

N_VELOCITY_POINTS = 40
# Kept away from very-low-speed/high-rpm corners, where alpha runs deep
# into stall and the resulting hard-to-converge solve can take much longer
# per point. SOLVE_TIMEOUT_S is a backstop against any point that's still slow.
V_FRACTION_RANGE = (0.5, 1.3)                     # of the loaded design-point velocity
RPM_FRACTIONS = [0.90, 0.95, 1.00, 1.05, 1.10]    # of the loaded design-point rpm
SOLVE_TIMEOUT_S = 5.0

# Sequential blue, light -> dark: RPM lines are an ordered quantity (pilot
# cadence), not unrelated categories.
RPM_LINE_COLORS = ["#9ec5f4", "#5598e7", "#256abf", "#184f95", "#0d366b"]
WARNING_COLOR = "#fab219"   # reserved for the trusted-envelope band, never a series color
SURFACE = "#fcfcfb"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
MUTED_TEXT = "#898781"
PRIMARY_TEXT = "#0b0b0b"


def _in_envelope(results):
    """True if every station's alpha/analysis_confidence stayed inside
    the same envelope the optimizer itself enforced at the design point.
    """
    for sec in results["sections"]:
        alpha = float(sec["alpha_deg"])
        confidence = float(sec["analysis_confidence"])
        if not (ALPHA_MIN_DEG <= alpha <= ALPHA_MAX_DEG) or confidence < MIN_ANALYSIS_CONFIDENCE:
            return False
    return True


def build_performance_map(csv_filepath=DEFAULT_CSV):
    prop, design_rpm, design_velocity, rho, mu = load_prop_csv(csv_filepath)

    v_min = V_FRACTION_RANGE[0] * design_velocity
    v_max = V_FRACTION_RANGE[1] * design_velocity
    velocities = np.linspace(v_min, v_max, N_VELOCITY_POINTS)
    rpms = [f * design_rpm for f in RPM_FRACTIONS]

    data = {rpm: {"thrust": [], "power": [], "efficiency": [], "in_envelope": []} for rpm in rpms}

    for v in velocities:
        # Velocity is baked into BEMAnalysis's CasADi graph at construction
        # (unlike rpm/Omega, which is a runtime rootfinder argument) -- so
        # build the graph ONCE per velocity and reuse it across all RPM
        # lines by just mutating .omega, instead of rebuilding per (v, rpm).
        analysis = BEMAnalysis(propeller=prop, rpm=rpms[0], velocity=float(v), rho=rho, mu=mu)
        for rpm in rpms:
            analysis.omega = rpm * 2 * np.pi / 60
            try:
                # Fresh, single-use executor per point, shut down with
                # wait=False: bounds how long one slow solve can block the
                # sweep without leaving a stuck point queued behind others.
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(analysis.run)
                try:
                    results = future.result(timeout=SOLVE_TIMEOUT_S)
                finally:
                    executor.shutdown(wait=False)
                thrust = float(results["thrust"])
                power = float(results["power"])
                efficiency = float(results["efficiency"])
                ok = _in_envelope(results)
            except FutureTimeoutError:
                print(f"  v={float(v):.2f} m/s, rpm={rpm:.0f}: solve exceeded "
                      f"{SOLVE_TIMEOUT_S:.0f}s -- skipped")
                thrust = power = efficiency = float("nan")
                ok = False
            except Exception:
                thrust = power = efficiency = float("nan")
                ok = False
            data[rpm]["thrust"].append(thrust)
            data[rpm]["power"].append(power)
            data[rpm]["efficiency"].append(efficiency)
            data[rpm]["in_envelope"].append(ok)

    return velocities, rpms, data, design_rpm, design_velocity


def plot_performance_map(velocities, rpms, data, design_rpm, design_velocity,
                          save_path=DEFAULT_PLOT_PATH):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    quantities = [("thrust", "Thrust (N)"), ("power", "Power (W)"), ("efficiency", "Efficiency (-)")]

    # "In envelope" is the rare case here -- only a narrow band near the
    # design point -- so shade that trusted band instead of marking every
    # untrusted point (which would bury the chart).
    closest_rpm = min(rpms, key=lambda r: abs(r - design_rpm))
    ref_env = data[closest_rpm]["in_envelope"]
    trusted_v = [v for v, ok in zip(velocities, ref_env) if ok]

    for ax, (key, ylabel) in zip(axes, quantities):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color(AXIS_COLOR)
        ax.spines["bottom"].set_color(AXIS_COLOR)
        ax.tick_params(colors=MUTED_TEXT)
        ax.set_ylabel(ylabel, color=PRIMARY_TEXT)
        ax.axvline(design_velocity, color=AXIS_COLOR, linewidth=1, linestyle="--")
        if trusted_v:
            ax.axvspan(min(trusted_v), max(trusted_v), color=WARNING_COLOR, alpha=0.10, zorder=0)

        for rpm, color in zip(rpms, RPM_LINE_COLORS):
            y = data[rpm][key]
            ax.plot(velocities, y, color=color, linewidth=2, label=f"{rpm:.0f} rpm")

    axes[-1].set_xlabel("Velocity (m/s)", color=PRIMARY_TEXT)
    axes[0].legend(loc="upper right", frameon=False, labelcolor=PRIMARY_TEXT, fontsize=9)
    axes[0].set_title(
        f"Propeller performance map  (design point: {design_velocity:.2f} m/s, {design_rpm:.1f} rpm, "
        f"dashed line)\n"
        f"Shaded band = where the {closest_rpm:.0f} rpm line stays inside the optimizer's own "
        f"alpha/confidence envelope -- NeuralFoil is\nprogressively less trustworthy the further "
        f"outside it you go",
        color=PRIMARY_TEXT, fontsize=10, loc="left",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=SURFACE)
    print(f"Saved: {save_path}")
    try:
        plt.show()
    except Exception:
        pass  # no display available -- the saved PNG is still there


def print_summary(velocities, rpms, data, design_rpm, design_velocity):
    print(f"Design point: {design_velocity:.2f} m/s @ {design_rpm:.1f} rpm")
    print(f"Velocity sweep: {velocities[0]:.2f} - {velocities[-1]:.2f} m/s "
          f"({V_FRACTION_RANGE[0]:.0%}-{V_FRACTION_RANGE[1]:.0%} of design velocity)")
    print(f"RPM lines: {', '.join(f'{r:.0f}' for r in rpms)}")
    print()
    # Print the design-rpm row in full, as a quick console-readable table.
    closest_rpm = min(rpms, key=lambda r: abs(r - design_rpm))
    print(f"Detail at {closest_rpm:.0f} rpm (closest sweep line to the design rpm):")
    print(f"{'V (m/s)':>8} {'Thrust (N)':>11} {'Power (W)':>10} {'Efficiency':>11} {'In envelope?':>13}")
    d = data[closest_rpm]
    for i, v in enumerate(velocities):
        flag = "yes" if d["in_envelope"][i] else "NO"
        print(f"{v:8.2f} {d['thrust'][i]:11.2f} {d['power'][i]:10.2f} {d['efficiency'][i]:11.4f} {flag:>13}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    velocities, rpms, data, design_rpm, design_velocity = build_performance_map(csv_path)
    print_summary(velocities, rpms, data, design_rpm, design_velocity)
    plot_performance_map(velocities, rpms, data, design_rpm, design_velocity)
