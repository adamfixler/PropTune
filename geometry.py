import csv

import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt


class Propeller:

    def __init__(
        self,
        radius,
        hub_radius,
        chord,
        twist,
        airfoil: asb.Airfoil,
        n_blades=2,
        round_sections=None,
    ):

        self.radius = radius
        self.hub_radius = hub_radius

        self.chord = chord
        self.twist = twist          # Stored in DEGREES
        self.airfoil = asb.Airfoil(airfoil)
        self.n_blades = n_blades

        self.n_stations = chord.shape[0] if hasattr(chord, "shape") else len(chord)

        # True where a station is a round section (chord = its diameter)
        # instead of the airfoil. All False = the airfoil everywhere.
        self.round_sections = (np.zeros(self.n_stations, dtype=bool) if round_sections is None
                               else np.asarray(round_sections, dtype=bool))

        # Station centers, not node endpoints. A station placed exactly at
        # r=radius would zero the Prandtl tip-loss factor there and
        # degenerate the BEM residual, letting the optimizer exploit it.
        # Segment midpoints keep every station strictly inside
        # (hub_radius, radius).
        nodes = np.linspace(hub_radius, radius, self.n_stations + 1)
        self.r = 0.5 * (nodes[:-1] + nodes[1:])
        # Per-station segment width (array, not a single scalar) so BEM.py
        # can index it per station regardless of which seed built the prop.
        self.dr = np.diff(nodes)


def export_airfoil_dat(airfoil: asb.Airfoil, filepath, name=None):
    """Export airfoil coordinates as a Selig-format .dat file (name header,
    then x,y pairs from trailing edge over the top to the leading edge and
    back along the bottom).

    Import this into XFLR5's airfoil database first (Direct Foil Design ->
    File -> Open, or drag-and-drop the .dat) under the exact same name as
    the Airfoil object passed to to_wing()/export via
    asb.Airplane.export_XFLR5_xml() -- XFLR5 plane files reference airfoils
    by name, they don't embed coordinates, so the wing XML is unusable in
    XFLR5 until this airfoil exists in its database under a matching name.
    """
    name = name or airfoil.name
    coords = airfoil.coordinates
    with open(filepath, "w") as f:
        f.write(f"{name}\n")
        for x, y in coords:
            f.write(f"  {float(x):.6f}  {float(y):.6f}\n")


def export_qblade_bld(prop, filepath, name="Optimized_HPA_Prop",
                       polar_filename="dae51_polar.plr"):
    """Export the blade geometry as a QBlade .bld file (2.0.9.x format).

    Covers geometry only (POS/CHORD/TWIST/offsets). QBlade also needs a
    .plr polar file per station, covering the full 360-degree alpha range
    -- `polar_filename` here is a PLACEHOLDER. To generate a real one:
      1. Import the airfoil .dat (see export_airfoil_dat) into QBlade's
         Airfoil module.
      2. Run QBlade's Direct Analysis (XFoil-linked) at your design's
         Reynolds numbers, then Polar Extrapolation (Viterna) for full
         360-degree coverage.
      3. Either rename that polar to match `polar_filename`, or re-enter
         these station values directly into QBlade's Blade Design table.

    Twist is measured about the leading edge (x/c=0), matching this
    project's to_wing() convention -- hence TAXIS=0.0 for every station.
    """
    import datetime
    now = datetime.datetime.now()

    lines = [
        "----------------------------------------QBlade Blade Definition File------------------------------------------------",
        "Generated with : AeroSandbox HPA propeller optimizer (export_qblade_bld)",
        "Archive Format: 310002",
        f"Time : {now.strftime('%H:%M:%S')}",
        f"Date : {now.strftime('%d.%m.%Y')}",
        "----------------------------------------Object Name-----------------------------------------------------------------",
        f"{name} OBJECTNAME - the name of the blade object",
        "----------------------------------------Parameters------------------------------------------------------------------",
        "HAWT ROTORTYPE - the rotor type",
        f"{prop.n_blades} NUMBLADES - number of blades",
        "----------------------------------------Blade Data------------------------------------------------------------------",
        "POS [m]  CHORD [m]  TWIST [deg]  OFFSET_X [m]  OFFSET_Y [m]  TAXIS [-]  POLAR_FILE",
    ]

    for i in range(prop.n_stations):
        r = float(prop.r[i])
        c = float(prop.chord[i])
        t = float(prop.twist[i])
        lines.append(f"{r:.4f}  {c:.4f}  {t:.4f}  0.0000  0.0000  0.0000  {polar_filename}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


def export_prop_csv(filepath, r, dr, chord, twist, radius, hub_radius, n_blades,
                     airfoil_name, rpm, velocity, rho, mu, round_sections=None):
    """Export a propeller's per-station geometry and its operating point/
    environment to a plain CSV, so a standalone script (see
    verify_xfoil.py) can re-load the exact design without re-running the
    optimization. Metadata (everything that isn't per-station) is written
    as '# key,value' comment lines above the station table;
    load_prop_csv() reads both halves back out.

    r, dr, chord, twist must already be plain numbers (e.g. sol.value(...)
    results), not CasADi opti variables.
    """
    metadata = [
        ("radius_m", radius),
        ("hub_radius_m", hub_radius),
        ("n_blades", n_blades),
        ("airfoil", airfoil_name),
        ("rpm", rpm),
        ("velocity_mps", velocity),
        ("rho", rho),
        ("mu", mu),
    ]
    with open(filepath, "w", newline="") as f:
        for key, value in metadata:
            f.write(f"# {key},{value}\n")
        writer = csv.writer(f)
        writer.writerow(["station", "r_m", "dr_m", "chord_m", "twist_deg", "round"])
        for i in range(len(r)):
            is_round = int(bool(round_sections[i])) if round_sections is not None else 0
            writer.writerow([i, float(r[i]), float(dr[i]), float(chord[i]), float(twist[i]), is_round])


def to_wing(prop):

    xsecs = []

    for i in range(prop.n_stations):

        xsecs.append(

            asb.WingXSec(

                xyz_le=[0, float(prop.r[i]), 0],

                chord=prop.chord[i],

                twist=prop.twist[i],

                airfoil=prop.airfoil,

            )
        )

    return asb.Wing(
        xsecs=xsecs,
        symmetric=False,
    )


def export_blade_step(prop, filepath, minimum_te_thickness=0.001, root_embed=0.025,
                      hub_top_z=None, hub_length=None, center_on_hub=False,
                      sections_per_station=4):
    """Export the blade as a watertight STEP solid, suitable for CFD.

    Builds each cross-section as a single unsplit spline (AeroSandbox's
    own export_cadquery_geometry splits at the leading edge, which leaves
    the loft unstitched there), so the loft closes into one genuine solid.
    Reuses AeroSandbox's per-station transform, so the blade shape itself
    is unchanged.

    The blade spans hub to tip, positioned about the shaft axis (origin),
    so it can be imported and attached to a hub of radius hub_radius
    WITHOUT moving it radially -- any radial shift moves every section
    off its design radius and changes the aerodynamics.

    Tip: a section at r=radius, with chord and twist linearly
    extrapolated from the last two stations, so the taper continues
    (holding the last section constant leaves a visible kink where the
    taper stops). Over the last BEM strip the mean chord stays ~= the
    last station's, as BEM assumes.
    Between stations: chord and twist are interpolated monotonically
    (PCHIP), sections_per_station evenly spaced sections per station
    spacing, and the blade is lofted through all of them in one smooth
    loft. Every station's section is reproduced exactly.
    Root: the first station's section is extruded straight (constant
    chord and twist) to root_embed (m) inside the hub, so it buries into
    the hub. Extruded rather than lofted -- the smooth loft overshoots a
    sudden switch to constant chord.

    If hub_top_z and hub_length (m) are given, also works out the
    z-translation that centres the blade on a hub spanning z = hub_top_z -
    hub_length ... hub_top_z (see hub_placement()), prints it, and returns
    it. With center_on_hub=True that translation is applied to the STEP
    itself, so it imports already in place. Otherwise the blade's leading
    edge stays on the y-axis and the translation must be applied in CAD.

    Relies on AeroSandbox's internal Wing._compute_frame_of_WingXSec() --
    may need updating if a future AeroSandbox version changes that method.
    """
    import cadquery as cq
    from scipy.interpolate import PchipInterpolator

    r = np.array(prop.r, dtype=float)
    chord = np.array(prop.chord, dtype=float)
    twist = np.array(prop.twist, dtype=float)
    weight = np.where(prop.round_sections, 0.0, 1.0)  # 1 = airfoil, 0 = round section
    if prop.radius > r[-1]:
        frac = (prop.radius - r[-1]) / (r[-1] - r[-2])
        r = np.append(r, prop.radius)
        # Floor stops a steep taper extrapolating to a zero/negative chord.
        chord = np.append(chord, max(chord[-1] + frac * (chord[-1] - chord[-2]), 0.5 * chord[-1]))
        twist = np.append(twist, twist[-1] + frac * (twist[-1] - twist[-2]))
        weight = np.append(weight, 1.0)

    # Round root: a circle (diameter = the round station's chord), centred on
    # the airfoil's thickest point at the first airfoil station (the spar),
    # blending linearly into the airfoil by that station.
    if prop.round_sections.any():
        i_af = int(np.argmin(prop.round_sections))  # first airfoil station
        x_t = np.linspace(0, 1, 501)
        x_tmax = x_t[np.argmax(prop.airfoil.local_thickness(x_t))]
        round_d = chord[i_af - 1]
        round_centre = np.array([x_tmax, float(prop.airfoil.local_camber(x_tmax))]) * chord[i_af]

    # Loft through evenly spaced sections (uneven spacing makes the loft
    # overshoot), with chord/twist between stations from a monotone (PCHIP)
    # interpolation. A loft through the stations alone overshoots wherever
    # the taper changes slope sharply (a dip-then-bulge in chord near a
    # thickness-constrained tip); with 4 sections per station spacing the
    # surface stays within ~0.3 mm of the intended chord.
    step = np.min(np.diff(prop.r)) / sections_per_station if len(prop.r) > 1 else r[-1] - r[0]
    r_loft = np.linspace(r[0], r[-1], int(round((r[-1] - r[0]) / step)) + 1)
    chord_loft = PchipInterpolator(r, chord)(r_loft)
    twist_loft = PchipInterpolator(r, twist)(r_loft)
    weight_loft = np.interp(r_loft, r, weight)
    wing = asb.Wing(xsecs=[
        asb.WingXSec(xyz_le=[0, float(ri), 0], chord=float(ci), twist=float(ti), airfoil=prop.airfoil)
        for ri, ci, ti in zip(r_loft, chord_loft, twist_loft)
    ], symmetric=False)

    def section(i, wing=wing):
        xsec = wing.xsecs[i]
        csys = wing._compute_frame_of_WingXSec(i)
        af = xsec.airfoil
        if af.TE_thickness() < minimum_te_thickness:
            af = af.set_TE_thickness(thickness=minimum_te_thickness)
        points = af.coordinates * xsec.chord
        w = weight_loft[i]
        if w < 1:
            # Each airfoil point maps to the circle point at the same fraction
            # of the perimeter, so every section keeps the same point order.
            seg = np.linalg.norm(np.diff(np.vstack([af.coordinates, af.coordinates[:1]]), axis=0), axis=1)
            theta = 2 * np.pi * np.concatenate([[0], np.cumsum(seg[:-1])]) / np.sum(seg)
            circle = round_centre + 0.5 * round_d * np.column_stack([np.cos(theta), np.sin(theta)])
            points = w * points + (1 - w) * circle

        # Normal is -span (-y), so a positive extrude() runs toward the hub.
        workplane = cq.Workplane(
            inPlane=cq.Plane(
                origin=tuple(xsec.xyz_le),
                xDir=tuple(csys[0]),
                normal=tuple(-csys[1]),
            )
        )
        return workplane.spline(
            listOfXYTuple=[tuple(xy) for xy in points]
        ).close()

    # Smooth loft, built directly rather than via CadQuery's loft(): that
    # leaves OpenCASCADE's wire-compatibility check on, which re-matches
    # points between the twisted sections and makes the surface wobble
    # (chord errors of ~15 mm near the tip). Every section is the same
    # point list in the same order, so they're already compatible.
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    builder = BRepOffsetAPI_ThruSections(True, False, 1e-6)  # solid, smooth
    for i in range(len(wing.xsecs)):
        builder.AddWire(section(i).wires().val().wrapped)
    builder.CheckCompatibility(False)
    builder.Build()
    loft = cq.Workplane("XY").newObject([cq.Solid(builder.Shape())])

    root_ext = float(prop.r[0]) - (prop.hub_radius - root_embed)
    if root_ext > 0:
        # Small fuzzy tolerance: without it the union fails on a round root.
        loft = loft.union(section(0).extrude(root_ext), tol=1e-5)

    solids = loft.solids().vals()
    if len(solids) != 1 or not loft.val().isValid():
        raise RuntimeError(
            f"export_blade_step: result is not a single valid watertight "
            f"solid (solids={len(solids)}, valid={loft.val().isValid()}) -- "
            f"refusing to export a STEP file that CFD tools would reject."
        )

    solid = loft.val()
    placement = None
    if hub_top_z is not None and hub_length is not None:
        placement = hub_placement(solid, hub_top_z, hub_length)
        if center_on_hub:
            solid = solid.translate(cq.Vector(0, 0, placement["dz"]))
        print_hub_placement(placement, applied=center_on_hub)

    cq.exporters.export(solid.scale(1000), str(filepath))  # STEP files are conventionally in mm
    return placement


def hub_placement(blade_solid, hub_top_z, hub_length):
    """The z-translation that centres a blade solid (in metres, shaft axis =
    z through the origin) axially on a hub spanning z = hub_top_z -
    hub_length ... hub_top_z (m). Only z is needed: the blade already sits
    on the hub axis. Returns a dict of the placement (all in m).
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(blade_solid.wrapped, box, False, False)  # tight, not the padded default
    z_min, z_max = box.Get()[2], box.Get()[5]

    hub_bottom_z = hub_top_z - hub_length
    dz = 0.5 * (hub_bottom_z + hub_top_z) - 0.5 * (z_min + z_max)
    return {
        "hub_bottom_z": hub_bottom_z, "hub_top_z": hub_top_z,
        "z_min": z_min, "z_max": z_max, "dz": dz,
        "clearance": 0.5 * (hub_length - (z_max - z_min)),
    }


def print_hub_placement(p, applied=False):
    print("\nHub placement (shaft axis = z through the origin; do NOT move in x or y):")
    print(f"  hub                z = {p['hub_bottom_z']:+.5f} .. {p['hub_top_z']:+.5f} m")
    print(f"  blade as lofted    z = {p['z_min']:+.5f} .. {p['z_max']:+.5f} m  "
          f"(axial extent {p['z_max'] - p['z_min']:.5f} m)")
    if applied:
        print(f"  already centred in the STEP (moved by {p['dz'] * 1000:+.2f} mm in z) "
              f"-- import it as-is, no translation needed")
    else:
        print(f"  translate blade by (0, 0, {p['dz']:+.5f}) m  =  (0, 0, {p['dz'] * 1000:+.2f}) mm")
    print(f"  blade on the hub   z = {p['z_min'] + p['dz']:+.5f} .. {p['z_max'] + p['dz']:+.5f} m")
    if p["clearance"] >= 0:
        print(f"  clearance to each hub face: {p['clearance'] * 1000:.2f} mm")
    else:
        print(f"  WARNING: blade is {-2 * p['clearance'] * 1000:.2f} mm longer axially than the "
              f"hub -- it will overhang both hub faces by {-p['clearance'] * 1000:.2f} mm")


def export_cfd_params(filepath, params, comment="PropTune design parameters for CFD setup"):
    """Write a flat key=value file (Java .properties format) describing the
    exported design -- STEP file, RPM, hub placement, predictions -- so a
    CFD setup script (e.g. a STAR-CCM+ Java macro, via
    java.util.Properties.load) can apply a new design without hand edits.
    Paths are written with forward slashes (backslashes are escapes in
    .properties); booleans as true/false.
    """
    import datetime

    def fmt(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return repr(value)
        return str(value).replace("\\", "/")

    with open(filepath, "w") as f:
        f.write(f"# {comment}\n# generated {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        for key, value in params.items():
            f.write(f"{key}={fmt(value)}\n")


def display_rotor(prop):

    wing = to_wing(prop)

    wing.draw_three_view(
        style="wireframe",
        show=False,
    )
    plt.show()


def load_bem(filename, airfoil):

    with open(filename, "r") as f:
        lines = f.readlines()

    n_blades = None
    diameter = None
    table_start = None

    # -------------------------
    # Read header
    # -------------------------

    for i, line in enumerate(lines):

        line = line.strip()

        if line.startswith("Num_Blade"):
            n_blades = int(line.split(":")[1])

        elif line.startswith("Diameter"):
            diameter = float(line.split(":")[1])

        elif line.startswith("Radius/R"):
            table_start = i + 1
            break

    if n_blades is None:
        raise RuntimeError("Couldn't find Num_Blade in .bem file.")

    if diameter is None:
        raise RuntimeError("Couldn't find Diameter in .bem file.")

    if table_start is None:
        raise RuntimeError("Couldn't find geometry table in .bem file.")

    R = diameter / 2

    # -------------------------
    # Read node data
    # -------------------------

    r_nodes = []
    chord_nodes = []
    twist_nodes = []

    for line in lines[table_start:]:

        line = line.strip()

        if not line:
            continue

        vals = line.replace(",", " ").split()

        try:
            vals = [float(v) for v in vals]
        except ValueError:
            continue

        if len(vals) < 3:
            continue

        r = vals[0] * R
        c = vals[1] * R
        beta = vals[2]

        # Ignore dummy tip points
        if c <= 1e-6:
            continue

        r_nodes.append(r)
        chord_nodes.append(c)
        twist_nodes.append(beta)

    r_nodes = np.array(r_nodes)
    chord_nodes = np.array(chord_nodes)
    twist_nodes = np.array(twist_nodes)

    # -------------------------
    # Convert node values to
    # element-centred values
    # -------------------------

    r = 0.5 * (r_nodes[:-1] + r_nodes[1:])
    chord = 0.5 * (chord_nodes[:-1] + chord_nodes[1:])
    twist = 0.5 * (twist_nodes[:-1] + twist_nodes[1:])

    prop = Propeller(
        radius=R,
        hub_radius=r[0],
        chord=chord,
        twist=twist,
        airfoil=airfoil,
        n_blades=n_blades,
    )

    # Replace automatically generated stations with imported ones
    prop.r = r
    prop.n_stations = len(r)

    # One dr per station (matches len(r)), not a single averaged scalar --
    # BEM.py indexes this per station for each station's own integral.
    prop.dr = np.diff(r_nodes)

    return prop


def load_prop_csv(filepath):
    """Load a propeller design exported by export_prop_csv(), returning
    (prop, rpm, velocity, rho, mu) ready to hand straight to BEMAnalysis
    (or verify_xfoil's XFoil-based check). Reconstructs a Propeller with
    the exact station radii/segment widths that were actually analyzed/
    optimized, not Propeller.__init__'s default re-gridding.
    """
    with open(filepath, "r", newline="") as f:
        lines = f.readlines()

    metadata = {}
    table_start = None
    for i, line in enumerate(lines):
        if line.startswith("#"):
            key, _, value = line[1:].strip().partition(",")
            metadata[key.strip()] = value.strip()
        else:
            table_start = i
            break

    if table_start is None:
        raise RuntimeError(f"No station table found in {filepath!r}.")

    rows = list(csv.DictReader(lines[table_start:]))
    if not rows:
        raise RuntimeError(f"Station table in {filepath!r} is empty.")

    r = np.array([float(row["r_m"]) for row in rows])
    dr = np.array([float(row["dr_m"]) for row in rows])
    chord = np.array([float(row["chord_m"]) for row in rows])
    twist = np.array([float(row["twist_deg"]) for row in rows])
    round_sections = [row.get("round") == "1" for row in rows]  # column absent in older CSVs

    prop = Propeller(
        radius=float(metadata["radius_m"]),
        hub_radius=float(metadata["hub_radius_m"]),
        chord=chord,
        twist=twist,
        airfoil=metadata["airfoil"],
        n_blades=int(metadata["n_blades"]),
        round_sections=round_sections,
    )
    # Replace the default-regridded stations with the ones actually analyzed.
    prop.r = r
    prop.dr = dr
    prop.n_stations = len(r)

    rpm = float(metadata["rpm"])
    velocity = float(metadata["velocity_mps"])
    rho = float(metadata["rho"])
    mu = float(metadata["mu"])

    return prop, rpm, velocity, rho, mu


if __name__ == "__main__":
    prop = load_bem("deadelus_MIL_baseline.bem","dae51")
    display_rotor(prop)
