"""Calibrate the BEM model against CFD: compare per-station thrust/torque
from a CFD run of an exported blade with BEM's prediction for the same
design, then fit per-station Cl/Cd corrections that make BEM reproduce the
CFD loads. The corrections are written to a file that main.py applies when
USE_CFD_CORRECTIONS = True.

Inputs:
- DESIGN_CSV: the design that was run in CFD, in export_prop_csv() format
  (geometry + operating point). Keep a copy outside outputs/ -- main.py
  overwrites outputs/optimized_prop.csv on every run.
- CFD per-face exports of the BLADE surface only (e.g. a STAR-CCM+ XYZ
  Internal Table on the blade part), columns "r_f" (face radius from the
  shaft axis), "Fz_p"/"Fz_s" (axial pressure/shear force on the face) and
  "Mz_f" (face moment about the shaft axis). Several exports (snapshots
  at different iterations) are averaged.

The correction: at each station, an additive dCl and dCd on NeuralFoil's
coefficients, solved so that BEM's station thrust AND torque both match
CFD exactly (two equations, two unknowns per station). Applied inside the
induction solve, so the inflow stays consistent with the corrected loads.
Stored against r/R so it can be reused on a different design of similar
size -- re-run CFD on the re-optimized design to check it still holds.

Usage: python cfd_calibration.py [design.csv] [export1.csv export2.csv ...]
"""
import csv
import datetime
import glob
import os
import sys

import aerosandbox.numpy as np
import matplotlib.pyplot as plt

from geometry import load_prop_csv
from BEM import BEMAnalysis

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
CALIBRATION_DIR = "calibration"
DESIGN_CSV = os.path.join(CALIBRATION_DIR, "28N_dae51_design.csv")
CFD_EXPORT_GLOB = os.path.join("PropTableExports", "*BladeOnly.csv")

# CFD -> full propeller. Half-model (one of two blades) -> x2.
CFD_FORCE_MULTIPLIER = 2.0
# Signs that make thrust and torque positive. With the STEP export's axes
# (shaft = +z, flow along -z, blade turning about +z) thrust is +Fz and the
# fluid moment on the blade is negative (it resists the rotation).
THRUST_SIGN = +1.0
TORQUE_SIGN = -1.0

CORRECTIONS_FILE = os.path.join(CALIBRATION_DIR, "cfd_corrections.csv")
PLOT_PATH = os.path.join(CALIBRATION_DIR, "cfd_calibration.png")

# Correction fit (per-station Newton solve on dCl, dCd)
FIT_MAX_ITERS = 15
FIT_TOL_N = 1e-5               # N / N.m per station
MAX_STEP_DCL, MAX_STEP_DCD = 0.2, 0.02
# ==============================================================================

CFD_COLOR = "#2a78d6"          # blue   -- CFD
BEM_COLOR = "#eb6834"          # orange -- BEM (uncorrected)
CORR_COLOR = "#1baf7a"         # aqua   -- BEM with CFD corrections
SURFACE = "#fcfcfb"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
MUTED_TEXT = "#898781"
PRIMARY_TEXT = "#0b0b0b"


def load_cfd_export(path):
    """Per-face (r, thrust, torque) from one CFD export, full-propeller
    values (CFD_FORCE_MULTIPLIER and signs applied)."""
    with open(path) as f:
        header = [h.strip().strip('"') for h in f.readline().split(",")]
    col = {name: header.index(name) for name in ("r_f", "Fz_p", "Fz_s", "Mz_f")}
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    r = d[:, col["r_f"]]
    thrust = THRUST_SIGN * CFD_FORCE_MULTIPLIER * (d[:, col["Fz_p"]] + d[:, col["Fz_s"]])
    torque = TORQUE_SIGN * CFD_FORCE_MULTIPLIER * d[:, col["Mz_f"]]
    return r, thrust, torque


def station_edges(prop):
    """Strip edges around each BEM station; first/last widened to catch
    every face (blade root faces at the hub, tip faces at r ~ radius)."""
    edges = np.concatenate([prop.r - prop.dr / 2, [prop.r[-1] + prop.dr[-1] / 2]])
    edges[0], edges[-1] = 0.0, np.inf
    return edges


def bin_by_station(r, values, edges):
    idx = np.digitize(r, edges) - 1
    return np.bincount(idx, weights=values, minlength=len(edges) - 1)[: len(edges) - 1]


def run_bem(analysis, dcl, dcd):
    analysis.cl_offset, analysis.cd_offset = dcl, dcd
    res = analysis.run()
    secs = res["sections"]
    return (np.array([float(s["dT"]) for s in secs]),
            np.array([float(s["dQ"]) for s in secs]),
            secs)


def fit_corrections(analysis, T_cfd, Q_cfd):
    """Per-station (dCl, dCd) so BEM's station thrust and torque match CFD.

    Stations are independent in BEM (each has its own induction solve), so
    the Jacobian is block-diagonal: three BEM runs per iteration (base,
    dCl-perturbed, dCd-perturbed) give every station's 2x2 Jacobian at once.
    """
    n = len(T_cfd)
    dcl, dcd = np.zeros(n), np.zeros(n)
    h_cl, h_cd = 1e-3, 1e-4
    for it in range(FIT_MAX_ITERS):
        T, Q, _ = run_bem(analysis, dcl, dcd)
        fT, fQ = T - T_cfd, Q - Q_cfd
        worst = max(np.abs(fT).max(), np.abs(fQ).max())
        print(f"  fit iteration {it}: worst station mismatch {worst:.2e}")
        if worst < FIT_TOL_N:
            return dcl, dcd, True
        T1, Q1, _ = run_bem(analysis, dcl + h_cl, dcd)
        T2, Q2, _ = run_bem(analysis, dcl, dcd + h_cd)
        a, b = (T1 - T) / h_cl, (T2 - T) / h_cd     # dT/dCl, dT/dCd
        c, d = (Q1 - Q) / h_cl, (Q2 - Q) / h_cd     # dQ/dCl, dQ/dCd
        det = a * d - b * c
        step_cl = -(d * fT - b * fQ) / det
        step_cd = -(-c * fT + a * fQ) / det
        dcl = dcl + np.clip(step_cl, -MAX_STEP_DCL, MAX_STEP_DCL)
        dcd = dcd + np.clip(step_cd, -MAX_STEP_DCD, MAX_STEP_DCD)
    return dcl, dcd, False


def write_corrections(path, prop, rpm, velocity, design_csv, exports, dcl, dcd,
                      T_cfd, Q_cfd, T_bem, Q_bem, secs):
    metadata = [
        ("created", datetime.date.today().isoformat()),
        ("design_csv", design_csv),
        ("cfd_exports", ";".join(os.path.basename(e) for e in exports)),
        ("radius_m", prop.radius),
        ("hub_radius_m", prop.hub_radius),
        ("airfoil", prop.airfoil.name),
        ("rpm", rpm),
        ("velocity_mps", velocity),
        ("note", "dCl/dCd are additive to NeuralFoil Cl/Cd; interpolated by r/R in main.py"),
    ]
    with open(path, "w", newline="") as f:
        for key, value in metadata:
            f.write(f"# {key},{value}\n")
        w = csv.writer(f)
        w.writerow(["station", "r_over_R", "r_m", "dCl", "dCd", "alpha_deg", "Re",
                    "cfd_dT_N", "bem_dT_N", "cfd_dQ_Nm", "bem_dQ_Nm"])
        for i in range(prop.n_stations):
            w.writerow([i, float(prop.r[i] / prop.radius), float(prop.r[i]),
                        float(dcl[i]), float(dcd[i]),
                        float(secs[i]["alpha_deg"]), float(secs[i]["Re"]),
                        float(T_cfd[i]), float(T_bem[i]), float(Q_cfd[i]), float(Q_bem[i])])


def load_cfd_corrections(path):
    """Read a corrections file written by this script. Returns
    (r_over_R, dCl, dCd, metadata)."""
    with open(path, newline="") as f:
        lines = f.readlines()
    metadata = {}
    start = 0
    for start, line in enumerate(lines):
        if not line.startswith("#"):
            break
        key, _, value = line[1:].strip().partition(",")
        metadata[key.strip()] = value.strip()
    rows = list(csv.DictReader(lines[start:]))
    r_over_R = np.array([float(row["r_over_R"]) for row in rows])
    dcl = np.array([float(row["dCl"]) for row in rows])
    dcd = np.array([float(row["dCd"]) for row in rows])
    return r_over_R, dcl, dcd, metadata


def interpolate_corrections(path, r_stations, radius):
    """dCl, dCd at the given station radii (interpolated by r/R; held
    constant beyond the calibrated range), plus the file's metadata."""
    r_over_R, dcl, dcd, metadata = load_cfd_corrections(path)
    x = np.asarray(r_stations) / radius
    return np.interp(x, r_over_R, dcl), np.interp(x, r_over_R, dcd), metadata


def _style(ax, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED_TEXT)
    ax.set_ylabel(ylabel, color=PRIMARY_TEXT)
    ax.set_title(title, color=PRIMARY_TEXT, fontsize=10, loc="left")


def plot(prop, r, T_snaps, Q_snaps, T_cfd, Q_cfd, T_bem, Q_bem, T_cor, Q_cor,
         dcl, dcd, totals_rows, save_path):
    dr = prop.dr
    fig = plt.figure(figsize=(12, 13))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 1, 0.55])

    for col, (name, symbol, unit, snaps, cfd, bem, cor) in enumerate([
        ("Thrust", "T", "N/m", T_snaps, T_cfd, T_bem, T_cor),
        ("Torque", "Q", "N.m/m", Q_snaps, Q_cfd, Q_bem, Q_cor),
    ]):
        # Loading per unit span, so stations of different width compare fairly.
        ax = fig.add_subplot(gs[0, col])
        _style(ax, f"d{symbol}/dr ({unit})", f"{name} distribution (full propeller)")
        if len(snaps) > 1:
            lo, hi = np.min(snaps, axis=0) / dr, np.max(snaps, axis=0) / dr
            ax.fill_between(r, lo, hi, color=CFD_COLOR, alpha=0.18, linewidth=0,
                            label="CFD snapshot spread")
        ax.plot(r, cfd / dr, color=CFD_COLOR, linewidth=2, marker="o", markersize=4, label="CFD")
        ax.plot(r, bem / dr, color=BEM_COLOR, linewidth=2, marker="o", markersize=4, label="BEM")
        ax.plot(r, cor / dr, color=CORR_COLOR, linewidth=2, linestyle="--",
                label="BEM + CFD corrections")
        ax.legend(loc="upper left", frameon=False, labelcolor=PRIMARY_TEXT, fontsize=9)

        # CFD/BEM per station. Inboard stations where BEM predicts ~zero load
        # give meaningless ratios -- clipped to the axis, values in the table.
        ax = fig.add_subplot(gs[1, col])
        _style(ax, "CFD / BEM", f"{name} ratio per station (1 = BEM matches CFD)")
        ratio = cfd / bem
        on_scale = (ratio >= 0.6) & (ratio <= 1.4)
        ax.axhline(1.0, color=AXIS_COLOR, linewidth=1, linestyle="--")
        ax.plot(r[on_scale], ratio[on_scale], color=CFD_COLOR, linewidth=2, marker="o", markersize=4)
        ax.set_ylim(0.6, 1.4)
        ax.set_xlim(fig.axes[-2].get_xlim())
        if not on_scale.all():
            ax.text(0.98, 0.95, f"{(~on_scale).sum()} inboard station(s) off scale\n"
                    "(BEM load ~ 0 there) -- see console table",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED_TEXT)

    for col, (vals, label) in enumerate([(dcl, "dCl"), (dcd, "dCd")]):
        ax = fig.add_subplot(gs[2, col])
        _style(ax, label, f"Fitted correction {label} (added to NeuralFoil {label[1:]})")
        ax.axhline(0.0, color=AXIS_COLOR, linewidth=1)
        ax.plot(r, vals, color=CORR_COLOR, linewidth=2, marker="o", markersize=4)
        ax.set_xlabel("r (m)", color=PRIMARY_TEXT)

    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    table = ax.table(cellText=[row[1:] for row in totals_rows[1:]],
                     rowLabels=[row[0] for row in totals_rows[1:]],
                     colLabels=totals_rows[0][1:], loc="center", cellLoc="right")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for cell in table.get_celld().values():
        cell.set_edgecolor(GRID_COLOR)
        cell.get_text().set_color(PRIMARY_TEXT)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor=SURFACE)
    print(f"Saved: {save_path}")
    try:
        plt.show()
    except Exception:
        pass  # no display available -- the saved PNG is still there


def main(design_csv, exports):
    if not exports:
        sys.exit(f"No CFD exports found (looked for {CFD_EXPORT_GLOB}).")
    prop, rpm, velocity, rho, mu = load_prop_csv(design_csv)
    omega = rpm * 2 * np.pi / 60
    edges = station_edges(prop)
    r = np.asarray(prop.r)

    print(f"Design: {design_csv}  ({prop.airfoil.name}, R={prop.radius} m, {rpm:.2f} rpm, {velocity} m/s)")
    T_snaps, Q_snaps = [], []
    for path in exports:
        rf, t, q = load_cfd_export(path)
        T_snaps.append(bin_by_station(rf, t, edges))
        Q_snaps.append(bin_by_station(rf, q, edges))
        print(f"  {os.path.basename(path)}: {len(rf)} faces, r {rf.min():.4f}..{rf.max():.4f} m, "
              f"T={T_snaps[-1].sum():.3f} N, Q={Q_snaps[-1].sum():.3f} N.m")
        if rf.max() < 0.98 * prop.radius:
            print(f"  WARNING: blade only reaches r={rf.max():.4f} m, design radius is "
                  f"{prop.radius} m -- was the blade moved radially?")
    T_snaps, Q_snaps = np.array(T_snaps), np.array(Q_snaps)
    T_cfd, Q_cfd = T_snaps.mean(axis=0), Q_snaps.mean(axis=0)
    if T_cfd.sum() <= 0 or Q_cfd.sum() <= 0:
        print("WARNING: CFD thrust or torque is not positive -- check THRUST_SIGN/TORQUE_SIGN "
              "and the rotation/flow direction in the CFD setup.")

    analysis = BEMAnalysis(propeller=prop, rpm=rpm, velocity=velocity, rho=rho, mu=mu)
    n = prop.n_stations
    T_bem, Q_bem, secs = run_bem(analysis, np.zeros(n), np.zeros(n))

    print("\nFitting per-station Cl/Cd corrections:")
    dcl, dcd, ok = fit_corrections(analysis, T_cfd, Q_cfd)
    if not ok:
        print("WARNING: correction fit did not fully converge -- check the station table.")
    T_cor, Q_cor, _ = run_bem(analysis, dcl, dcd)

    def totals(T, Q):
        P = Q.sum() * omega
        return T.sum(), Q.sum(), P, T.sum() * velocity / P

    rows = [["", "Thrust (N)", "Torque (N.m)", "Power (W)", "Efficiency"]]
    for label, T, Q in ([(f"CFD {os.path.basename(e)}", t, q) for e, t, q in zip(exports, T_snaps, Q_snaps)]
                        + [("CFD mean", T_cfd, Q_cfd), ("BEM", T_bem, Q_bem),
                           ("BEM + corrections", T_cor, Q_cor)]):
        t, q, p, e = totals(T, Q)
        rows.append([label, f"{t:.3f}", f"{q:.3f}", f"{p:.1f}", f"{e:.4f}"])
    tc, qc, _, ec = totals(T_cfd, Q_cfd)
    tb, qb, _, eb = totals(T_bem, Q_bem)
    rows.append(["CFD / BEM", f"{tc / tb:.4f}", f"{qc / qb:.4f}", f"{qc / qb:.4f}", f"{ec / eb:.4f}"])

    print("\nTotals (full propeller, blade only):")
    for row in rows:
        print(f"  {row[0]:28s}" + "".join(f"{v:>14s}" for v in row[1:]))

    spread = (T_snaps.max(axis=0) - T_snaps.min(axis=0)) / np.abs(T_cfd) * 100 if len(exports) > 1 else np.zeros(n)
    print(f"\n{'st':>3} {'r (m)':>6} {'r/R':>5} {'alpha':>6} {'Re/1e3':>6} | {'CFD dT':>7} {'BEM dT':>7} {'T ratio':>7} "
          f"{'snap%':>5} | {'CFD dQ':>7} {'BEM dQ':>7} {'Q ratio':>7} | {'dCl':>7} {'dCd':>8}")
    for i in range(n):
        print(f"{i:3d} {r[i]:6.3f} {r[i] / prop.radius:5.3f} {float(secs[i]['alpha_deg']):6.2f} "
              f"{float(secs[i]['Re']) / 1e3:6.0f} | {T_cfd[i]:7.3f} {T_bem[i]:7.3f} {T_cfd[i] / T_bem[i]:7.3f} "
              f"{spread[i]:5.1f} | {Q_cfd[i]:7.3f} {Q_bem[i]:7.3f} {Q_cfd[i] / Q_bem[i]:7.3f} | "
              f"{dcl[i]:+7.4f} {dcd[i]:+8.5f}")
    deficit = T_bem - T_cfd
    for lo, hi in [(0, 0.35), (0.35, 0.7), (0.7, 1.01)]:
        m = (r / prop.radius >= lo) & (r / prop.radius < hi)
        print(f"  BEM - CFD thrust from r/R {lo:.2f}-{min(hi, 1):.2f}: {deficit[m].sum():+.3f} N")

    os.makedirs(os.path.dirname(CORRECTIONS_FILE) or ".", exist_ok=True)
    write_corrections(CORRECTIONS_FILE, prop, rpm, velocity, design_csv, exports, dcl, dcd,
                      T_cfd, Q_cfd, T_bem, Q_bem, secs)
    print(f"\nSaved: {CORRECTIONS_FILE}  (set USE_CFD_CORRECTIONS = True in main.py to use it)")

    plot(prop, r, T_snaps, Q_snaps, T_cfd, Q_cfd, T_bem, Q_bem, T_cor, Q_cor, dcl, dcd, rows, PLOT_PATH)


if __name__ == "__main__":
    design = sys.argv[1] if len(sys.argv) > 1 else DESIGN_CSV
    exports = sys.argv[2:] if len(sys.argv) > 2 else sorted(glob.glob(CFD_EXPORT_GLOB))
    main(design, exports)
