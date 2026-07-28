#Define Prop class, blade parameterisation, Spline interpolation, Blade stations
import aerosandbox as asb
import aerosandbox.numpy as np
import time
import matplotlib.pyplot as plt
opti = asb.Opti()  # an optimization environment (CasADi + IPOPT under the hood)

class Propeller:
    def __init__(
        self,
        radius,
        hub_radius,
        chord,
        twist,
        airfoil,
    ):
        self.radius = radius
        self.hub_radius = hub_radius
        self.chord = chord
        self.twist = twist
        self.airfoil = airfoil


airfoil = asb.Airfoil("dae51")
N = 10

#twists = opti.variable(init_guess=np.zeros(N))
#chords = opti.variable(init_guess=np.ones(N))

#twists = np.zeros(N)
#chords = np.ones(N)


twists = np.linspace(35, 5, N)
chords = np.linspace(0.15, 0.05, N)

prop = Propeller(
    radius=1.7,
    hub_radius=0.1,
    chord=chords,
    twist=twists,
    airfoil=airfoil,
)




def to_wing(prop):

    r = np.linspace(prop.hub_radius, prop.radius, N)

    xsecs = []

    for i in range(N):
        xsecs.append(
            asb.WingXSec(
                xyz_le=[0, float(r[i]), 0],
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
    wing.draw_three_view(style="wireframe", show=False)
    plt.show()

display_rotor(prop)