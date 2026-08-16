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

        # Station centers, NOT node endpoints. A station placed exactly at
        # r=radius makes the Prandtl tip-loss factor F -> 0 *exactly*, which
        # degenerates the BEM residual equation to 0==0 there -- it stops
        # constraining a/a' at all, and a gradient-based optimizer will
        # happily dump nonphysical induction values into that one station
        # to fake extra efficiency. Using segment midpoints (matching what
        # load_bem() already does) keeps every station strictly inside
        # (hub_radius, radius).
        nodes = np.linspace(hub_radius, radius, self.n_stations + 1)
        self.r = 0.5 * (nodes[:-1] + nodes[1:])
        # Per-station segment width, not a single scalar -- keeps this
        # array-shaped like load_bem()'s (potentially non-uniform) dr, so
        # BEM.py can index it per station regardless of which seed built
        # the Propeller.
        self.dr = np.diff(nodes)


def export_airfoil_dat(airfoil: asb.Airfoil, filepath, name=None):
    """Export airfoil coordinates as a Selig-format .dat file (name header,
    then x,y pairs from trailing edge over the top to the leading edge and
    back along the bottom).

    Import this into XFLR5's airfoil database FIRST (Direct Foil Design ->
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
    """Export the blade geometry as a QBlade .bld file (confirmed real
    format, from QBlade's own documentation -- 2.0.9.x series).

    IMPORTANT: this only covers geometry (POS/CHORD/TWIST/offsets). QBlade
    also needs a .plr polar file per station, covering the FULL 360 degree
    alpha range (via QBlade's own Viterna/Montgomery extrapolation) -- this
    project's NeuralFoil-based BEM never computed that (we deliberately
    stayed in a narrow, well-attached alpha band), so `polar_filename` here
    is a PLACEHOLDER. Generate the real polar inside QBlade itself:
      1. Import the airfoil .dat (see export_airfoil_dat) into QBlade's
         Airfoil module.
      2. Run QBlade's Direct Analysis (XFoil-linked) at your design's
         Reynolds numbers, then use Polar Extrapolation (Viterna) to get
         full 360-degree coverage.
      3. Either rename that polar to match `polar_filename`, or just
         re-enter these station values directly into QBlade's Blade
         Design table (only ~15 rows) and pick the real polar there.

    Twist here is measured about the leading edge (x/c=0), matching this
    project's to_wing() convention -- hence TAXIS=0.0 for every station,
    not the mid-chord/quarter-chord value you'd see in some example files.
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

    # Variable element spacing (more general than assuming uniform spacing).
    # One dr per station (len(r_nodes)-1 == len(r)), not a single averaged
    # scalar -- BEM.py indexes this per station so each station's true
    # segment width is used in its own thrust/torque integral.
    prop.dr = np.diff(r_nodes)

    return prop

if __name__ == "__main__":
    prop = load_bem("deadelus_MIL_baseline.bem","dae51")
    display_rotor(prop)
#prop = Propeller(hub_radius=0.1,radius=3, chord=np.ones(5), twist=np.zeros(5),airfoil="dae51",n_blades=2)