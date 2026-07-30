import aerosandbox as asb
import aerosandbox.numpy as np

from geometry import *
from BEM import *

opti = asb.Opti()

airfoil = "dae51"
T_required = 90
N = 10




prop = load_bem(
    "deadelus_MIL_baseline.bem",
    airfoil="dae51",
)

chord = opti.variable(
    init_guess=prop.chord,
    lower_bound=0.01,
    upper_bound=0.35,
)

twist = opti.variable(
    init_guess=prop.twist,
    lower_bound=-5,
    upper_bound=90,
)

analysis = BEMAnalysis(
    propeller=prop,
    rpm=180,
    velocity=10.95248,
)

results = analysis.run()

print(f"Thrust     : {results['thrust'][0]:.2f} N")
print(f"Torque     : {results['torque'][0]:.2f} Nm")
print(f"Power      : {results['power'][0]:.2f} W")
print(f"Efficiency : {results['efficiency'][0]:.3f}")

for sec in results["sections"]:
    print(
        f"r={sec['r']:.2f} "
        f"alpha={sec['alpha_deg']:.2f}° "
        f"Cl={sec['Cl'][0]:.3f}"
    )


def optimise():
    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    opti.subject_to(results["thrust"] >= T_required)
    
    # Keep every station in attached flow. This isn't just a numerical nicety --
    # an efficient prop should never run near CLmax -- but it also keeps
    # NeuralFoil out of its high-curvature stall-blending region, which is what
    # causes IPOPT's Hessian to blow up with "Invalid_Number_Detected".
    for sec in results["sections"]:
        opti.subject_to(sec["alpha_deg"] <= 10)
        opti.subject_to(sec["alpha_deg"] >= -4)
    
    # ------------------------------------------------------------------
    # Objective: maximize efficiency (Opti only exposes .minimize())
    # ------------------------------------------------------------------
    opti.minimize(-results["efficiency"])
    
    sol = opti.solve(max_iter=500)
    
    chord_opt = sol.value(chord)
    twist_opt = sol.value(twist)
    r = sol.value(prop.r)
    
    print(f"Efficiency : {sol.value(results['efficiency']):.4f}")
    print(f"Thrust     : {sol.value(results['thrust']):.2f} N")
    print(f"Torque     : {sol.value(results['torque']):.2f} Nm")
    print(f"Power      : {sol.value(results['power']):.2f} W")
    print()
    print(f"{'r (m)':>8} {'chord (m)':>10} {'twist (deg)':>12}")
    for ri, ci, ti in zip(r, chord_opt, twist_opt):
        print(f"{ri:8.3f} {ci:10.4f} {ti:12.2f}")

    prop_opt = Propeller(hub_radius=prop.hub_radius,radius=prop.radius,chord=chord_opt,twist=twist_opt,airfoil=airfoil,n_blades=2)
    display_rotor(prop_opt)

#optimise()
