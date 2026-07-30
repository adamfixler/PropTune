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

        self.n_stations = chord.shape[0]

        self.r = np.linspace(
            hub_radius,
            radius,
            self.n_stations
        )

        self.dr = (radius - hub_radius) / (self.n_stations - 1)


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

    # Variable element spacing (more general than assuming uniform spacing)
    dr = np.diff(r_nodes)
    prop.dr = np.mean(dr)

    return prop

#prop = load_bem("deadelus_MIL_baseline.bem","dae51")

#display_rotor(prop)
#prop = Propeller(hub_radius=0.1,radius=3, chord=np.ones(5), twist=np.zeros(5),airfoil="dae51",n_blades=2)