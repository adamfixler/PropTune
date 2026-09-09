import aerosandbox as asb
import aerosandbox.numpy as np

from geometry import Propeller, load_bem, display_rotor, export_airfoil_dat, export_qblade_bld, export_prop_csv, to_wing
from BEM import BEMAnalysis

# ==============================================================================
# USER CONFIGURATION -- every value you're likely to want to change lives here.
# Everything below this block is derived from these values, or is BEM/optimizer
# mechanics that shouldn't normally need touching.
# ==============================================================================

# --- Seed geometry source -----------------------------------------------------
# False (default): build an initial guess from published Daedalus dimensions
#   (11.3 ft / 3.44 m diameter, 8.3 in / 0.211 m max chord) plus a standard
#   BEM initial-guess technique (linear chord taper, twist = local flow
#   angle + a few degrees angle of attack).
# True: load station geometry from a real .bem file instead.
USE_BEM_FILE = False
BEM_FILENAME = "deadelus_MIL_baseline.bem"  # only used when USE_BEM_FILE=True

# --- Airfoil -------------------------------------------------------------------
airfoil = "dae51"

# --- Design point: English Channel crossing mission -----------------------------
velocity = 6.7      # m/s cruise
T_required = 28.0   # N required cruise thrust

# --- Radial station resolution ---------------------------------------------------
# NeuralFoil evaluates a large symbolic graph per station, un-batched, so
# higher N increases optimizer memory and solve time substantially. 10-20
# stations is standard BEM design resolution.
N = 20

# --- From-scratch seed geometry (USE_BEM_FILE=False only) ------------------------
# Real Daedalus numbers (MIT, web.mit.edu/drela/Public/web/hpa/dae_prop.pdf).
radius = 3.5 / 2               # m
hub_radius = 0.085             # m -- root cutout
n_blades = 2
max_chord = 8.3 * 0.0254       # 8.3 in -> m

# --- Design variable bounds -----------------------------------------------------
# rpm is a free design variable, optimized jointly with chord/twist. Bounds
# are a sane range for a human-powered aircraft (Daedalus cruised at ~108
# rpm). rpm_guess also seeds the from-scratch twist calculation below.
chord_min, chord_max = 0.01, 0.35   # m
twist_min, twist_max = -5, 90       # deg
rpm_guess = 110
rpm_min, rpm_max = 60, 220

# --- Constraints -----------------------------------------------------------------
alpha_min_deg, alpha_max_deg = -4, 10
# Minimum NeuralFoil analysis_confidence allowed per station -- keeps the
# optimizer inside NeuralFoil's trusted (alpha, Re) region.
min_analysis_confidence = 0.9

# --- Smoothness objective (penalizes jagged station-to-station chord/twist
#     jumps; see the objective section below for how these are used) -------------
smoothness_weight = 0.0025   # raise for a smoother/less jagged blade, lower to allow more shape freedom
twist_scale = 10.0           # deg -- normalizes the twist penalty against the chord penalty
# Hard per-segment rate bounds, in physical units (chord/twist change per
# meter of span), tied to manufacturing tolerance rather than an arbitrary
# dimensionless number.
max_chord_rate = 0.4    # m of chord change per m of span
max_twist_rate = 90.0   # deg of twist change per m of span

# --- Export filenames --------------------------------------------------------------
airfoil_dat_filename = "dae51.dat"
xflr5_xml_filename = "optimized_prop_blade.xml"
step_filename = "optimized_prop_blade.step"
qblade_bld_filename = "optimized_prop.bld"
qblade_polar_filename = "dae51_polar.plr"  # placeholder -- see export_qblade_bld()'s docstring
prop_csv_filename = "optimized_prop.csv"   # geometry + operating point, reloadable by verify_xfoil.py

# --- Post-optimization XFoil verification (optional) --------------------------------
# Re-checks the optimized blade using real XFoil instead of NeuralFoil, and
# prints it side-by-side against the NeuralFoil-based result -- see
# verify_xfoil.py. Requires the XFoil executable, and is much slower than
# the optimization above (XFoil runs as a subprocess per station, per
# iteration). Can also be run later, standalone: `python verify_xfoil.py`.
RUN_XFOIL_CHECK = True
XFOIL_COMMAND = r"C:\Users\adamf\Downloads\XFOIL6.99\xfoil.exe"  # path to your xfoil executable

# ==============================================================================
# End of user configuration.
# ==============================================================================

if USE_BEM_FILE:
    baseline = load_bem(BEM_FILENAME, airfoil=airfoil)
    radius = baseline.radius
    hub_radius = baseline.hub_radius
    n_blades = baseline.n_blades

    r_stations = np.linspace(baseline.r[0], baseline.r[-1], N)
    # dr per station, array-shaped to match load_bem()'s spacing convention.
    dr = np.full(N, r_stations[1] - r_stations[0])
    chord_guess = np.interp(r_stations, baseline.r, baseline.chord)
    twist_guess = np.interp(r_stations, baseline.r, baseline.twist)

else:
    # Station centers, not node endpoints -- a station placed exactly at
    # r=hub_radius or r=radius would zero the Prandtl loss factor there and
    # degenerate that station's BEM residual, letting the optimizer exploit it.
    nodes = np.linspace(hub_radius, radius, N + 1)
    r_stations = 0.5 * (nodes[:-1] + nodes[1:])
    dr = np.diff(nodes)

    # Linear taper as an initial guess only -- the optimizer reshapes this.
    chord_guess = np.interp(
        r_stations,
        [hub_radius, 0.35 * radius, radius],
        [0.4 * max_chord, max_chord, 0.15 * max_chord],
    )

    # Twist guess: local flow angle (zero induction) plus a target angle of
    # attack -- standard BEM initial-guess technique.
    Omega_seed = rpm_guess * 2 * np.pi / 60
    phi0 = np.arctan2(velocity, Omega_seed * r_stations)
    twist_guess = np.degrees(phi0) + 4.0

opti = asb.Opti()

chord = opti.variable(init_guess=chord_guess, lower_bound=chord_min, upper_bound=chord_max)
twist = opti.variable(init_guess=twist_guess, lower_bound=twist_min, upper_bound=twist_max)
rpm = opti.variable(init_guess=rpm_guess, lower_bound=rpm_min, upper_bound=rpm_max)

prop = Propeller(radius=radius, hub_radius=hub_radius, chord=chord, twist=twist,
                  airfoil=airfoil, n_blades=n_blades)

# Propeller.__init__ assumes uniform station spacing; override with the
# actual station radii/segment widths so chord/twist are optimized at the
# same stations computed above.
prop.r = r_stations
prop.dr = dr
prop.n_stations = N

analysis = BEMAnalysis(propeller=prop, rpm=rpm, velocity=velocity)
results = analysis.run()

# ------------------------------------------------------------------
# Constraints
# ------------------------------------------------------------------
opti.subject_to(results["thrust"] == T_required)
opti.subject_to(results["efficiency"] <= 1.0)  # cheap physical sanity net

for sec in results["sections"]:
    opti.subject_to(sec["alpha_deg"] <= alpha_max_deg)
    opti.subject_to(sec["alpha_deg"] >= alpha_min_deg)
    opti.subject_to(sec["analysis_confidence"] >= min_analysis_confidence)

# ------------------------------------------------------------------
# Objective: maximize efficiency, softly penalized for station-to-station
# chord/twist jaggedness (not for deviating from any reference taper).
# ------------------------------------------------------------------
chord_scale = max_chord if not USE_BEM_FILE else np.max(chord_guess)

# Riemann-sum approximation of integral[(dchord/dr)^2 + (dtwist/dr)^2] dr,
# normalized by each quantity's characteristic scale and by dr itself so
# the result is independent of station count N.
dr_seg = r_stations[1:] - r_stations[:-1]
d_chord_dr = (chord[1:] - chord[:-1]) / dr_seg / chord_scale
d_twist_dr = (twist[1:] - twist[:-1]) / dr_seg / twist_scale
smoothness_penalty = np.sum((d_chord_dr ** 2 + d_twist_dr ** 2) * dr_seg)

# Hard per-segment bounds: guarantee no individual segment exceeds the
# limit, on top of the soft penalty above.
chord_rate = (chord[1:] - chord[:-1]) / dr_seg
twist_rate = (twist[1:] - twist[:-1]) / dr_seg
opti.subject_to(chord_rate <= max_chord_rate)
opti.subject_to(chord_rate >= -max_chord_rate)
opti.subject_to(twist_rate <= max_twist_rate)
opti.subject_to(twist_rate >= -max_twist_rate)

opti.minimize(-results["efficiency"] + smoothness_weight * smoothness_penalty)

sol = opti.solve(max_iter=1000)

chord_opt = sol.value(chord)
twist_opt = sol.value(twist)
rpm_opt = sol.value(rpm)

print(f"Efficiency : {sol.value(results['efficiency']):.4f}")
print(f"Smoothness : {sol.value(smoothness_penalty):.4f}  (weight={smoothness_weight})")
print(f"RPM        : {rpm_opt:.1f}")
print(f"Thrust     : {sol.value(results['thrust']):.2f} N")
print(f"Torque     : {sol.value(results['torque']):.2f} Nm")
print(f"Power      : {sol.value(results['power']):.2f} W")
print()
print(f"{'r (m)':>8} {'chord_0 (m)':>12} {'chord_opt (m)':>14} {'twist_0 (deg)':>14} {'twist_opt (deg)':>16}")
for ri, c0, ci, t0, ti in zip(r_stations, chord_guess, chord_opt, twist_guess, twist_opt):
    print(f"{ri:8.3f} {c0:12.4f} {ci:14.4f} {t0:14.2f} {ti:16.2f}")

# ------------------------------------------------------------------
# Display the optimized propeller (three-view wireframe)
# ------------------------------------------------------------------
prop_result = Propeller(
    radius=radius,
    hub_radius=hub_radius,
    chord=chord_opt,
    twist=twist_opt,
    airfoil=airfoil,
    n_blades=n_blades,
)
display_rotor(prop_result)

# ------------------------------------------------------------------
# Export for XFLR5 (AeroSandbox's native exporter). Import dae51.dat into
# XFLR5's airfoil database under the name "DAE51" before opening the plane
# XML -- XFLR5 references airfoils by name only, it doesn't embed coordinates.
#
# Note: XFLR5's LLT/VLM analysis treats this as a static wing, not a
# rotating blade -- it won't reproduce this project's BEM results.
# ------------------------------------------------------------------
export_airfoil_dat(prop_result.airfoil, airfoil_dat_filename, name=prop_result.airfoil.name)

blade_wing = to_wing(prop_result)
blade_wing.name = "Blade"
airplane = asb.Airplane(name="Optimized HPA Prop Blade", wings=[blade_wing])
airplane.export_XFLR5_xml(xflr5_xml_filename)

# ------------------------------------------------------------------
# CAD export (AeroSandbox's native exporter) -- a self-contained STEP
# solid, openable in Fusion 360, SolidWorks, FreeCAD, etc.
# ------------------------------------------------------------------
airplane.export_cadquery_geometry(step_filename)

print(f"\nExported: {airfoil_dat_filename}, {xflr5_xml_filename}, {step_filename}")

# ------------------------------------------------------------------
# Export for QBlade (geometry only). QBlade also needs a full 360-degree
# .plr polar per station -- generate one in QBlade itself (Direct Analysis
# + Viterna extrapolation) before this file is usable there; see
# export_qblade_bld()'s docstring.
# ------------------------------------------------------------------
export_qblade_bld(prop_result, qblade_bld_filename, polar_filename=qblade_polar_filename)
print(f"Exported: {qblade_bld_filename}  (edit POLAR_FILE once you've generated a real polar in QBlade)")

# ------------------------------------------------------------------
# Export geometry + operating point as CSV, so the design can be
# re-checked (e.g. against XFoil, see verify_xfoil.py) without re-running
# the optimizer.
# ------------------------------------------------------------------
export_prop_csv(
    prop_csv_filename,
    r=r_stations, dr=dr, chord=chord_opt, twist=twist_opt,
    radius=radius, hub_radius=hub_radius, n_blades=n_blades,
    airfoil_name=airfoil, rpm=rpm_opt, velocity=velocity,
    rho=analysis.rho, mu=analysis.mu,
)
print(f"Exported: {prop_csv_filename}  (feed this to verify_xfoil.py for a real-XFoil performance check)")

if RUN_XFOIL_CHECK:
    from verify_xfoil import run_xfoil_check
    run_xfoil_check(prop_csv_filename, xfoil_command=XFOIL_COMMAND)
