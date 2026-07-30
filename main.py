import aerosandbox as asb
import aerosandbox.numpy as np

from geometry import Propeller, load_bem, display_rotor, export_airfoil_dat, to_wing
from BEM import BEMAnalysis

# ------------------------------------------------------------------
# Starting geometry. USE_BEM_FILE toggles between two seed sources:
#
#   False (default): a principled, non-fabricated initial guess, built
#   from real published Daedalus numbers (11.3 ft / 3.44 m diameter,
#   8.3 in / 0.211 m max chord -- from MIT's own dae_prop.pdf spec
#   sheet) plus basic flow-angle physics for twist. Nothing here is
#   invented -- it's either a cited real number or a standard BEM
#   initial-guess technique (linear chord taper, phi + a few degrees AoA
#   for twist).
#
#   True: load a real .bem file (e.g. from a proper design tool or a
#   trusted source) and use its actual station geometry as the seed --
#   this is exactly what the uploaded deadelus_MIL_baseline.bem was
#   being used for, and that capability is kept for whenever you have
#   a genuine file to seed from.
# ------------------------------------------------------------------
USE_BEM_FILE = False
BEM_FILENAME = "deadelus_MIL_baseline.bem"

airfoil = "dae51"

# ------------------------------------------------------------------
# Actual design point: English Channel crossing mission
# ------------------------------------------------------------------
velocity = 7.5      # m/s cruise
T_required = 28.0   # N required cruise thrust

# The loaded file has 89 stations. BEMAnalysis calls NeuralFoil (a genuinely
# large symbolic graph -- hundreds of nodes per call) once per station,
# un-batched, so optimizing at full resolution builds an enormous combined
# graph and IPOPT's Hessian evaluation runs out of memory. 10-20 stations
# is standard BEM design resolution anyway.
N = 15

if USE_BEM_FILE:
    baseline = load_bem(BEM_FILENAME, airfoil=airfoil)
    radius = baseline.radius
    hub_radius = baseline.hub_radius
    n_blades = baseline.n_blades

    r_stations = np.linspace(baseline.r[0], baseline.r[-1], N)
    dr = r_stations[1] - r_stations[0]
    chord_guess = np.interp(r_stations, baseline.r, baseline.chord)
    twist_guess = np.interp(r_stations, baseline.r, baseline.twist)

else:
    # Real Daedalus numbers (MIT, web.mit.edu/drela/Public/web/hpa/dae_prop.pdf):
    # 11.3 ft (3.44 m) diameter, 8.3 in (0.211 m) max chord, 2 blades.
    radius = 3.44 / 2       # m
    hub_radius = 0.1 * radius  # standard small root cutout, not from any file
    n_blades = 2
    max_chord = 8.3 * 0.0254   # 8.3 in -> m

    r_stations = np.linspace(hub_radius, radius, N)
    dr = r_stations[1] - r_stations[0]

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
    rpm_guess_for_seed = 110  # ballpark only, used just to seed twist
    Omega_seed = rpm_guess_for_seed * 2 * np.pi / 60
    phi0 = np.arctan2(velocity, Omega_seed * r_stations)
    twist_guess = np.degrees(phi0) + 4.0

opti = asb.Opti()

# ------------------------------------------------------------------
# Design variables: chord and twist at each station, AND rpm. rpm was
# never grounded in anything real (the .bem file has no design-condition
# metadata at all) -- rather than hand-pick a replacement, let the
# optimizer find the best rpm jointly with the blade geometry. Bounds
# are just a sane HPA range (real Daedalus cruised at 108 rpm).
# ------------------------------------------------------------------
chord = opti.variable(init_guess=chord_guess, lower_bound=0.01, upper_bound=0.35)
twist = opti.variable(init_guess=twist_guess, lower_bound=-5, upper_bound=90)
rpm = opti.variable(init_guess=110, lower_bound=60, upper_bound=220)

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
    opti.subject_to(sec["alpha_deg"] <= 10)
    opti.subject_to(sec["alpha_deg"] >= -4)
    # NeuralFoil reports its own analysis_confidence alongside CL/CD, from
    # the same differentiable computation graph -- surveyed across this
    # design's actual (alpha, Re) envelope, confidence is consistently
    # 0.94-0.99 for alpha >= -2 deg, but drops to 0.43-0.82 at alpha=-4 deg
    # (worse at low Re) -- meaning the alpha bound above was letting the
    # optimizer use a region NeuralFoil itself doesn't trust much. This
    # constraint keeps every station in the region NeuralFoil is actually
    # confident in, tightening things exactly where the alpha bound alone
    # wasn't catching it, without hand-picking a stricter alpha cutoff.
    opti.subject_to(sec["analysis_confidence"] >= 0.9)

# ------------------------------------------------------------------
# Objective: maximize efficiency, softly penalized for jaggedness
# (station-to-station chord/twist jumps). No fixed target shape to
# deviate from -- this only penalizes RAGGEDNESS itself, not distance
# from any particular reference blade, so it doesn't bias the design
# toward Daedalus's or anyone else's specific taper.
#
# Squared first differences between adjacent stations, normalized by a
# characteristic scale for each quantity so the two penalties are
# comparable and the weight below has a consistent meaning regardless
# of units. A soft penalty (vs. hard opti.subject_to bounds on the
# slope) means jaggedness costs something rather than being forbidden
# outright -- the optimizer will still pay for a sharp step if the
# efficiency gain genuinely justifies it, so it can't make the problem
# infeasible or silently forbid a shape you actually want.
# ------------------------------------------------------------------
chord_scale = max_chord if not USE_BEM_FILE else np.max(chord_guess)
twist_scale = 10.0  # degrees -- a "reasonable" per-station twist step

d_chord = (chord[1:] - chord[:-1]) / chord_scale
d_twist = (twist[1:] - twist[:-1]) / twist_scale
smoothness_penalty = np.sum(d_chord ** 2 + d_twist ** 2) / (N - 1)

smoothness_weight = 0.3  # raise for a smoother/less jagged blade, lower to allow more shape freedom

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
export_airfoil_dat(prop_result.airfoil, "dae51.dat", name=prop_result.airfoil.name)

blade_wing = to_wing(prop_result)
blade_wing.name = "Blade"
airplane = asb.Airplane(name="Optimized HPA Prop Blade", wings=[blade_wing])
airplane.export_XFLR5_xml("optimized_prop_blade.xml")

# ------------------------------------------------------------------
# CAD export, also via AeroSandbox's native exporter rather than a
# hand-rolled generator (asb.Airplane.export_cadquery_geometry). Produces
# a real solid-body STEP file, openable in any CAD package (Fusion 360,
# SolidWorks, FreeCAD, etc.) -- this one's self-contained, unlike the
# XFLR5 export, since STEP embeds the full airfoil geometry directly
# rather than referencing it by name from an external database.
# ------------------------------------------------------------------
airplane.export_cadquery_geometry("optimized_prop_blade.step")

print("\nExported: dae51.dat, optimized_prop_blade.xml, optimized_prop_blade.step")