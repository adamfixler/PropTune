import aerosandbox as asb
import aerosandbox.numpy as np

from geometry import Propeller, load_bem, display_rotor, export_airfoil_dat, export_qblade_bld, to_wing
from BEM import BEMAnalysis

# ==============================================================================
# USER CONFIGURATION -- every value you're likely to want to change lives here.
# Everything below this block is derived from these values, or is BEM/optimizer
# mechanics that shouldn't normally need touching.
# ==============================================================================

# --- Seed geometry source -----------------------------------------------------
# False (default): a principled, non-fabricated initial guess, built from real
#   published Daedalus numbers (11.3 ft / 3.44 m diameter, 8.3 in / 0.211 m max
#   chord -- from MIT's own dae_prop.pdf spec sheet) plus basic flow-angle
#   physics for twist. Nothing here is invented -- it's either a cited real
#   number or a standard BEM initial-guess technique (linear chord taper,
#   phi + a few degrees AoA for twist).
# True: load a real .bem file (e.g. from a proper design tool or a trusted
#   source) and use its actual station geometry as the seed -- this is exactly
#   what the uploaded deadelus_MIL_baseline.bem was being used for, and that
#   capability is kept for whenever you have a genuine file to seed from.
USE_BEM_FILE = False
BEM_FILENAME = "deadelus_MIL_baseline.bem"  # only used when USE_BEM_FILE=True

# --- Airfoil -------------------------------------------------------------------
airfoil = "dae51"

# --- Design point: English Channel crossing mission -----------------------------
velocity = 7.5      # m/s cruise
T_required = 28.0   # N required cruise thrust

# --- Radial station resolution ---------------------------------------------------
# The loaded file has 89 stations. BEMAnalysis calls NeuralFoil (a genuinely
# large symbolic graph -- hundreds of nodes per call) once per station,
# un-batched, so optimizing at full resolution builds an enormous combined
# graph and IPOPT's Hessian evaluation runs out of memory. 10-20 stations
# is standard BEM design resolution anyway.
N = 10

# --- From-scratch seed geometry (USE_BEM_FILE=False only) ------------------------
# Real Daedalus numbers (MIT, web.mit.edu/drela/Public/web/hpa/dae_prop.pdf):
# 11.3 ft (3.44 m) diameter, 8.3 in (0.211 m) max chord, 2 blades.
radius = 3.44 / 2              # m
hub_radius = 0.1 * radius      # m -- standard small root cutout, not from any file
n_blades = 2
max_chord = 8.3 * 0.0254       # 8.3 in -> m

# --- Design variable bounds -----------------------------------------------------
# rpm was never grounded in anything real (the .bem file has no
# design-condition metadata at all) -- rather than hand-pick a replacement,
# let the optimizer find the best rpm jointly with the blade geometry. Bounds
# below are just a sane HPA range (real Daedalus cruised at 108 rpm).
# rpm_guess is a ballpark starting point only -- it's used both as the
# optimizer's initial guess for the free rpm variable below, AND (in the
# from-scratch seed branch) to compute a plausible starting twist curve, so
# both uses stay in sync from one number instead of two that could drift
# apart for no reason.
chord_min, chord_max = 0.01, 0.35   # m
twist_min, twist_max = -5, 90       # deg
rpm_guess = 110
rpm_min, rpm_max = 60, 220

# --- Constraints -----------------------------------------------------------------
alpha_min_deg, alpha_max_deg = -4, 10
# NeuralFoil reports its own analysis_confidence alongside CL/CD, from the
# same differentiable computation graph -- surveyed across this design's
# actual (alpha, Re) envelope, confidence is consistently 0.94-0.99 for
# alpha >= -2 deg, but drops to 0.43-0.82 at alpha=-4 deg (worse at low Re)
# -- meaning the alpha bounds above alone were letting the optimizer use a
# region NeuralFoil itself doesn't trust much. This constraint keeps every
# station in the region NeuralFoil is actually confident in, tightening
# things exactly where the alpha bounds alone weren't catching it, without
# hand-picking a stricter alpha cutoff.
min_analysis_confidence = 0.9

# --- Smoothness objective (penalizes jagged station-to-station chord/twist
#     jumps; see the objective section below for how these are used) -------------
smoothness_weight = 0.0025   # raise for a smoother/less jagged blade, lower to allow more shape freedom
twist_scale = 10.0           # deg -- a "reasonable" per-station twist step, used to normalize the twist penalty against the chord penalty
# Hard per-segment rate bounds, in real physical units (max chord/twist
# change per meter of span) so they're tied to something meaningful
# (mold/layup/manufacturing tolerance) rather than being an arbitrary
# dimensionless number to tune by trial. Defaults set a bit above what the
# optimizer already found on its own (~0.30 m/m chord, ~80 deg/m twist) --
# loose enough not to fight the soft penalty above, but a real ceiling
# against any future degenerate configuration.
max_chord_rate = 0.4    # m of chord change per m of span
max_twist_rate = 90.0   # deg of twist change per m of span

# --- Export filenames --------------------------------------------------------------
airfoil_dat_filename = "dae51.dat"
xflr5_xml_filename = "optimized_prop_blade.xml"
step_filename = "optimized_prop_blade.step"
qblade_bld_filename = "optimized_prop.bld"
qblade_polar_filename = "dae51_polar.plr"  # placeholder -- see export_qblade_bld()'s docstring

# ==============================================================================
# End of user configuration.
# ==============================================================================

if USE_BEM_FILE:
    baseline = load_bem(BEM_FILENAME, airfoil=airfoil)
    radius = baseline.radius
    hub_radius = baseline.hub_radius
    n_blades = baseline.n_blades

    r_stations = np.linspace(baseline.r[0], baseline.r[-1], N)
    # One dr per station (uniform here, but array-shaped) so it indexes the
    # same way as the non-uniform dr load_bem() itself produces -- BEM.py
    # indexes prop.dr per station regardless of which seed built the prop.
    dr = np.full(N, r_stations[1] - r_stations[0])
    chord_guess = np.interp(r_stations, baseline.r, baseline.chord)
    twist_guess = np.interp(r_stations, baseline.r, baseline.twist)

else:
    # Station centers, NOT node endpoints -- mirrors Propeller.__init__'s
    # own midpoint construction. linspace(hub_radius, radius, N) would put
    # a station exactly at r=radius (and one at r=hub_radius), which zeros
    # the Prandtl tip-loss factor F there and degenerates that station's
    # BEM residual to 0==0, letting the optimizer dump nonphysical
    # induction values into it to fake extra efficiency. Every station
    # here stays strictly inside (hub_radius, radius).
    nodes = np.linspace(hub_radius, radius, N + 1)
    r_stations = 0.5 * (nodes[:-1] + nodes[1:])
    dr = np.diff(nodes)

    # Linear taper as an initial guess only -- the optimizer reshapes this.
    # Peaks near max_chord a bit outboard of the root, tapers to a modest
    # tip chord, which is the general shape any propeller blade takes.
    chord_guess = np.interp(
        r_stations,
        [hub_radius, 0.35 * radius, radius],
        [0.4 * max_chord, max_chord, 0.15 * max_chord],
    )

    # Twist guess: inflow angle at a starting rpm guess (zero induction),
    # plus a few degrees of target angle of attack -- standard BEM
    # initial-guess practice, not fabricated data.
    Omega_seed = rpm_guess * 2 * np.pi / 60
    phi0 = np.arctan2(velocity, Omega_seed * r_stations)
    twist_guess = np.degrees(phi0) + 4.0

opti = asb.Opti()

chord = opti.variable(init_guess=chord_guess, lower_bound=chord_min, upper_bound=chord_max)
twist = opti.variable(init_guess=twist_guess, lower_bound=twist_min, upper_bound=twist_max)
rpm = opti.variable(init_guess=rpm_guess, lower_bound=rpm_min, upper_bound=rpm_max)

prop = Propeller(radius=radius, hub_radius=hub_radius, chord=chord, twist=twist,
                  airfoil=airfoil, n_blades=n_blades)

# Propeller.__init__ assumes uniform station spacing by default; override
# with the baseline's actual station radii / segment width so we're
# optimizing chord/twist AT THE SAME RADIAL STATIONS as the loaded prop,
# not some arbitrary re-gridding of it.
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
# Objective: maximize efficiency, softly penalized for jaggedness
# (station-to-station chord/twist jumps). No fixed target shape to
# deviate from -- this only penalizes RAGGEDNESS itself, not distance
# from any particular reference blade, so it doesn't bias the design
# toward Daedalus's or anyone else's specific taper.
#
# Squared first differences between adjacent stations, normalized by a
# characteristic scale for each quantity so the two penalties are
# comparable and the weight above has a consistent meaning regardless
# of units. A soft penalty (vs. hard opti.subject_to bounds on the
# slope) means jaggedness costs something rather than being forbidden
# outright -- the optimizer will still pay for a sharp step if the
# efficiency gain genuinely justifies it, so it can't make the problem
# infeasible or silently forbid a shape you actually want.
# ------------------------------------------------------------------
chord_scale = max_chord if not USE_BEM_FILE else np.max(chord_guess)

# Riemann-sum approximation of integral[(dchord/dr)^2 + (dtwist/dr)^2] dr
# along the span, each normalized by a characteristic scale. Dividing by
# the actual radial spacing (not just the adjacent-index difference)
# makes this converge to the SAME value for a fixed underlying blade
# shape regardless of how many stations N you use -- the previous version
# used raw (chord[i+1]-chord[i]) with no dr normalization, so it shrank by
# roughly 1/N^2 as N grew (stations get physically closer together), and
# smoothness_weight quietly stopped doing anything at higher N.
dr_seg = r_stations[1:] - r_stations[:-1]
d_chord_dr = (chord[1:] - chord[:-1]) / dr_seg / chord_scale
d_twist_dr = (twist[1:] - twist[:-1]) / dr_seg / twist_scale
smoothness_penalty = np.sum((d_chord_dr ** 2 + d_twist_dr ** 2) * dr_seg)

# Unlike a hard bound on the aggregate smoothness_penalty sum, the rate
# bounds below guarantee NO individual segment exceeds the limit -- one bad
# transition can't hide behind otherwise-smooth neighbors.
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
# Export for XFLR5, using AeroSandbox's own native exporter rather than a
# hand-rolled XML generator (asb.Airplane.export_XFLR5_xml). Still need
# the separate airfoil .dat file, though: XFLR5 plane files reference
# airfoils by NAME only (this exporter uses xsec.airfoil.name), they
# don't embed coordinates -- import dae51.dat into XFLR5's own airfoil
# database under the name "DAE51" BEFORE opening the plane XML, or it
# won't be able to resolve the section airfoils.
#
# Also worth remembering: XFLR5's own LLT/VLM analysis on this wing
# treats it as a static lifting surface in uniform flow, not a rotating
# blade -- it won't reproduce the BEM results from this project.
# ------------------------------------------------------------------
export_airfoil_dat(prop_result.airfoil, airfoil_dat_filename, name=prop_result.airfoil.name)

blade_wing = to_wing(prop_result)
blade_wing.name = "Blade"
airplane = asb.Airplane(name="Optimized HPA Prop Blade", wings=[blade_wing])
airplane.export_XFLR5_xml(xflr5_xml_filename)

# ------------------------------------------------------------------
# CAD export, also via AeroSandbox's native exporter rather than a
# hand-rolled generator (asb.Airplane.export_cadquery_geometry). Produces
# a real solid-body STEP file, openable in any CAD package (Fusion 360,
# SolidWorks, FreeCAD, etc.) -- this one's self-contained, unlike the
# XFLR5 export, since STEP embeds the full airfoil geometry directly
# rather than referencing it by name from an external database.
# ------------------------------------------------------------------
airplane.export_cadquery_geometry(step_filename)

print(f"\nExported: {airfoil_dat_filename}, {xflr5_xml_filename}, {step_filename}")

# ------------------------------------------------------------------
# Export for QBlade. Same airfoil .dat file works for QBlade too (it
# accepts plain x,y .dat airfoil import directly). qblade_polar_filename
# here is a PLACEHOLDER -- see export_qblade_bld()'s docstring for the
# recommended workflow to generate a real, full-360-degree .plr polar
# inside QBlade itself before this blade definition is usable there.
# ------------------------------------------------------------------
export_qblade_bld(prop_result, qblade_bld_filename, polar_filename=qblade_polar_filename)
print(f"Exported: {qblade_bld_filename}  (edit POLAR_FILE once you've generated a real polar in QBlade)")
