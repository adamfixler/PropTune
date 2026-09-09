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
    ):

        self.radius = radius
        self.hub_radius = hub_radius

        self.chord = chord
        self.twist = twist          # Stored in DEGREES
        self.airfoil = asb.Airfoil(airfoil)
        self.n_blades = n_blades

        self.n_stations = chord.shape[0] if hasattr(chord, "shape") else len(chord)

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
                     airfoil_name, rpm, velocity, rho, mu):
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
        writer.writerow(["station", "r_m", "dr_m", "chord_m", "twist_deg"])
        for i in range(len(r)):
            writer.writerow([i, float(r[i]), float(dr[i]), float(chord[i]), float(twist[i])])


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

    prop = Propeller(
        radius=float(metadata["radius_m"]),
        hub_radius=float(metadata["hub_radius_m"]),
        chord=chord,
        twist=twist,
        airfoil=metadata["airfoil"],
        n_blades=int(metadata["n_blades"]),
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
