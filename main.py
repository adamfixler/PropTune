import aerosandbox as asb
import aerosandbox.numpy as np

from geometry import *
from BEM import *

airfoil = "dae51"

N = 10

twists = np.linspace(35, 5, N)      # degrees
chords = np.linspace(0.15, 0.05, N)


prop = load_bem(
    "deadelus_MIL_baseline.bem",
    airfoil="dae51",
)

#display_rotor(prop)


prop2 = Propeller(
    radius=1.7,
    hub_radius=0.1,
    chord=chords,
    twist=twists,
    airfoil=airfoil,
    n_blades=2,
)



analysis = BEMAnalysis(
    propeller=prop,
    rpm=180,
    velocity=10.95248,
)

results = analysis.run()

print(f"Thrust     : {results['thrust']:.2f} N")
print(f"Torque     : {results['torque']:.2f} Nm")
print(f"Power      : {results['power']:.2f} W")
print(f"Efficiency : {results['efficiency']:.3f}")

for sec in results["sections"]:
    print(
        f"r={sec['r']:.2f} "
        f"alpha={sec['alpha_deg']:.2f}° "
        f"Cl={sec['Cl']:.3f}"
    )