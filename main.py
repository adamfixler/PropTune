import os

import aerosandbox as asb
import aerosandbox.numpy as np

from geometry import Propeller, load_bem, display_rotor, export_airfoil_dat, export_qblade_bld, export_prop_csv, export_blade_step, export_cfd_params, to_wing
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
hub_radius = 0.0782 / 2        # m -- root cutout (hub diameter 78.2 mm)
n_blades = 2
max_chord = 8.3 * 0.0254       # 8.3 in -> m

# --- Hub placement in your CAD/CFD model ------------------------------------------
# The hub spans z = hub_top_z - hub_length ... hub_top_z (m), on the shaft
# (z) axis. Used to centre the exported STEP blade axially on the hub.
hub_top_z = 0.0                # m -- z of the hub's top face
hub_length = 0.30              # m -- hub axial length
# True: the STEP is exported already centred on the hub -- import it as-is.
# False: the STEP keeps the leading edge on the y-axis, and the z-translation
# to apply in CAD is printed instead.
CENTER_BLADE_ON_HUB = True

# --- Design variable bounds -----------------------------------------------------
# rpm is a free design variable, optimized jointly with chord/twist. Bounds
# are a sane range for a human-powered aircraft (Daedalus cruised at ~108
# rpm). rpm_guess also seeds the from-scratch twist calculation below.
chord_min, chord_max = 0.01, 0.20   # m -- chord_max: maximum chord along the entire span
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
max_chord_rate = 0.8    # m of chord change per m of span
max_twist_rate = 120   # deg of twist change per m of span

# --- Minimum airfoil thickness (max t/c x chord) ---------------------------------
min_root_thickness = 0.025      # m -- at the root station (carried into the hub by the export)
min_thickness = 0.005          # m -- everywhere else: every station and the exported tip
# Where the airfoil at chord_max is thinner than required, the station is a
# round section instead (diameter = the required thickness), blending into
# the airfoil by the next station.


# --- CFD corrections ------------------------------------------------------------------
# True: add the per-station Cl/Cd corrections fitted by cfd_calibration.py
# (from a CFD run of an earlier design) to NeuralFoil's coefficients, so the
# optimizer designs against CFD-calibrated loads. Corrections are
# interpolated by r/R onto this run's stations. False: plain BEM.
USE_CFD_CORRECTIONS = True
CFD_CORRECTIONS_FILE = os.path.join("calibration", "cfd_corrections.csv")

# --- Export filenames --------------------------------------------------------------
# All exports land in OUTPUT_DIR (created automatically if missing).
OUTPUT_DIR = "outputs"
airfoil_dat_filename = os.path.join(OUTPUT_DIR, "dae51.dat")
xflr5_xml_filename = os.path.join(OUTPUT_DIR, "optimized_prop_blade.xml")
step_filename = os.path.join(OUTPUT_DIR, "optimized_prop_blade.step")
qblade_bld_filename = os.path.join(OUTPUT_DIR, "optimized_prop.bld")
qblade_polar_filename = "dae51_polar.plr"  # placeholder, relative to qblade_bld_filename's own folder -- see export_qblade_bld()'s docstring
prop_csv_filename = os.path.join(OUTPUT_DIR, "optimized_prop.csv")   # geometry + operating point, reloadable by verify_xfoil.py
cfd_params_filename = os.path.join(OUTPUT_DIR, "cfd_params.properties")  # STEP file, rpm, hub placement -- for a CFD setup macro

# --- Post-optimization XFoil verification (optional) --------------------------------
# Re-checks the optimized blade using real XFoil instead of NeuralFoil, and
# prints it side-by-side against the NeuralFoil-based result -- see
# verify_xfoil.py. Requires the XFoil executable, and is much slower than
# the optimization above (XFoil runs as a subprocess per station, per
# iteration). Can also be run later, standalone: `python verify_xfoil.py`.
RUN_XFOIL_CHECK = False
XFOIL_COMMAND = r"C:\Path\To\XFOIL6.99\xfoil.exe"  # path to your xfoil executable

# ==============================================================================
# End of user configuration.
# ==============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Round sections: stations where the airfoil can't reach the required
# thickness within chord_max.
max_af_thickness = asb.Airfoil(airfoil).max_thickness()
t_required = np.full(N, min_thickness)
t_required[0] = min_root_thickness
round_sections = chord_max * max_af_thickness < t_required
chord_guess = np.where(round_sections, t_required, np.minimum(chord_guess, chord_max))

opti = asb.Opti()

chord = opti.variable(init_guess=chord_guess, lower_bound=chord_min, upper_bound=chord_max)
twist = opti.variable(init_guess=twist_guess, lower_bound=twist_min, upper_bound=twist_max)
rpm = opti.variable(init_guess=rpm_guess, lower_bound=rpm_min, upper_bound=rpm_max)

prop = Propeller(radius=radius, hub_radius=hub_radius, chord=chord, twist=twist,
                  airfoil=airfoil, n_blades=n_blades, round_sections=round_sections)

# Propeller.__init__ assumes uniform station spacing; override with the
# actual station radii/segment widths so chord/twist are optimized at the
# same stations computed above.
prop.r = r_stations
prop.dr = dr
prop.n_stations = N

cl_offset = cd_offset = None
if USE_CFD_CORRECTIONS:
    from cfd_calibration import interpolate_corrections
    cl_offset, cd_offset, corr_meta = interpolate_corrections(CFD_CORRECTIONS_FILE, r_stations, radius)
    print(f"Using CFD corrections from {CFD_CORRECTIONS_FILE} "
          f"(calibrated on {corr_meta.get('design_csv')}, {corr_meta.get('airfoil')}, "
          f"R={corr_meta.get('radius_m')} m, {float(corr_meta.get('rpm', 'nan')):.1f} rpm, "
          f"{corr_meta.get('velocity_mps')} m/s)")
    if corr_meta.get("airfoil") != airfoil:
        print(f"  WARNING: corrections were fitted for airfoil {corr_meta.get('airfoil')!r}, "
              f"this run uses {airfoil!r}")

analysis = BEMAnalysis(propeller=prop, rpm=rpm, velocity=velocity,
                       cl_offset=cl_offset, cd_offset=cd_offset)
results = analysis.run()

# ------------------------------------------------------------------
# Constraints
# ------------------------------------------------------------------
opti.subject_to(results["thrust"] == T_required)
opti.subject_to(results["efficiency"] <= 1.0)  # cheap physical sanity net

for sec, is_round in zip(results["sections"], round_sections):
    if is_round:  # no airfoil there: alpha and NeuralFoil confidence don't apply
        continue
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
# Chord changes between round sections and the airfoil are left free.
af_seg = ~(round_sections[1:] | round_sections[:-1])
d_chord_dr = (chord[1:] - chord[:-1]) / dr_seg / chord_scale * af_seg
d_twist_dr = (twist[1:] - twist[:-1]) / dr_seg / twist_scale
smoothness_penalty = np.sum((d_chord_dr ** 2 + d_twist_dr ** 2) * dr_seg)

# Hard per-segment bounds: guarantee no individual segment exceeds the
# limit, on top of the soft penalty above.
chord_rate = (chord[1:] - chord[:-1]) / dr_seg
twist_rate = (twist[1:] - twist[:-1]) / dr_seg
for k in np.flatnonzero(af_seg):
    opti.subject_to(chord_rate[k] <= max_chord_rate)
    opti.subject_to(chord_rate[k] >= -max_chord_rate)
opti.subject_to(twist_rate <= max_twist_rate)
opti.subject_to(twist_rate >= -max_twist_rate)

# Minimum thickness: at the root, at every station, and at the tip.
# Round sections are fixed at the required thickness (their diameter).
# The STEP export extends the blade from the last station out to r = radius,
# continuing the taper of the last two stations -- constrain that actual tip.
tip_frac = (radius - r_stations[-1]) / (r_stations[-1] - r_stations[-2])
tip_chord = chord[-1] + tip_frac * (chord[-1] - chord[-2])
tip_thickness = tip_chord * max_af_thickness

for i in range(N):
    if round_sections[i]:
        opti.subject_to(chord[i] == t_required[i])
    else:
        opti.subject_to(chord[i] * max_af_thickness >= t_required[i])
opti.subject_to(tip_thickness >= min_thickness)

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
print(f"{'r (m)':>8} {'chord_0 (m)':>12} {'chord_opt (m)':>14} {'twist_0 (deg)':>14} {'twist_opt (deg)':>16} {'thickness (mm)':>14}")
for ri, c0, ci, t0, ti, is_round in zip(r_stations, chord_guess, chord_opt, twist_guess, twist_opt, round_sections):
    t_mm = ci * 1000 if is_round else ci * max_af_thickness * 1000
    print(f"{ri:8.3f} {c0:12.4f} {ci:14.4f} {t0:14.2f} {ti:16.2f} {t_mm:14.4f}" + ("  round" if is_round else ""))
tip_chord_opt = chord_opt[-1] + tip_frac * (chord_opt[-1] - chord_opt[-2])
print(f"Tip (r = {radius} m): chord {tip_chord_opt:.4f} m, thickness {tip_chord_opt * max_af_thickness * 1000:.2f} mm")

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
    round_sections=round_sections,
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
# CAD export -- a self-contained, watertight STEP solid, openable in
# Fusion 360, SolidWorks, FreeCAD, etc., and suitable for CFD meshing.
# See export_blade_step()'s docstring in geometry.py for why this is
# used instead of AeroSandbox's own export_cadquery_geometry().
# ------------------------------------------------------------------
placement = export_blade_step(prop_result, step_filename, hub_top_z=hub_top_z,
                              hub_length=hub_length, center_on_hub=CENTER_BLADE_ON_HUB)

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
    rho=analysis.rho, mu=analysis.mu, round_sections=round_sections,
)
print(f"Exported: {prop_csv_filename}  (feed this to verify_xfoil.py for a real-XFoil performance check)")

# ------------------------------------------------------------------
# Design parameters for automating the CFD setup (e.g. a STAR-CCM+ macro
# that swaps in the new STEP and sets the rotation rate).
# ------------------------------------------------------------------
export_cfd_params(cfd_params_filename, {
    "step_file": os.path.abspath(step_filename),
    "step_centered_on_hub": CENTER_BLADE_ON_HUB,
    "blade_z_offset_m": 0.0 if CENTER_BLADE_ON_HUB else placement["dz"],  # still to apply in CAD
    "blade_z_min_m": placement["z_min"] + placement["dz"],                 # on the hub, once placed
    "blade_z_max_m": placement["z_max"] + placement["dz"],
    "hub_radius_m": float(hub_radius),
    "hub_top_z_m": float(hub_top_z),
    "hub_length_m": float(hub_length),
    "radius_m": float(radius),
    "n_blades": int(n_blades),
    "airfoil": airfoil,
    "rpm": float(rpm_opt),
    "omega_rad_s": float(rpm_opt * 2 * np.pi / 60),
    "velocity_mps": float(velocity),
    "rho": float(analysis.rho),
    "mu": float(analysis.mu),
    "cfd_corrections": CFD_CORRECTIONS_FILE if USE_CFD_CORRECTIONS else "none",
    "predicted_thrust_N": float(sol.value(results["thrust"])),
    "predicted_torque_Nm": float(sol.value(results["torque"])),
    "predicted_power_W": float(sol.value(results["power"])),
    "predicted_efficiency": float(sol.value(results["efficiency"])),
    "design_csv": os.path.abspath(prop_csv_filename),
})
print(f"Exported: {cfd_params_filename}  (STEP file, rpm and hub placement for CFD setup)")

if RUN_XFOIL_CHECK:
    from verify_xfoil import run_xfoil_check
    run_xfoil_check(prop_csv_filename, xfoil_command=XFOIL_COMMAND)
