import aerosandbox.numpy as np
import casadi as ca

from geometry import load_bem
from BEM import BEMAnalysis


def f(x):
    """Pull a plain Python float out of whatever BEMAnalysis returns
    (casadi DM/MX for a solved rootfinder output, or plain numpy)."""
    if isinstance(x, (ca.DM, ca.MX, ca.SX)):
        return float(ca.DM(x))
    return float(np.array(x).flatten()[0])


airfoil = "dae51"
baseline = load_bem("deadelus_MIL_baseline.bem", airfoil=airfoil)

rpm = 180
velocity = 10.95248  # m/s -- Daedalus MIL baseline reference cruise condition

analysis = BEMAnalysis(propeller=baseline, rpm=rpm, velocity=velocity)
results = analysis.run()

print(f"Baseline (Daedalus MIL), rpm={rpm}, V={velocity:.3f} m/s, "
      f"{baseline.n_stations} stations")
print(f"Thrust     : {f(results['thrust']):.2f} N")
print(f"Torque     : {f(results['torque']):.3f} Nm")
print(f"Power      : {f(results['power']):.2f} W")
print(f"Efficiency : {f(results['efficiency']):.4f}")
print()
print(f"{'r (m)':>8} {'chord (m)':>10} {'twist (deg)':>12} {'alpha (deg)':>12} {'Cl':>7} {'a':>7} {'ap':>7}")
for i, sec in enumerate(results["sections"]):
    print(f"{f(sec['r']):8.3f} {baseline.chord[i]:10.4f} "
          f"{f(sec['twist_deg']):12.2f} {f(sec['alpha_deg']):12.2f} "
          f"{f(sec['Cl']):7.3f} {f(sec['a']):7.4f} {f(sec['ap']):7.4f}")