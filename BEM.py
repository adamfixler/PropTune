import aerosandbox.numpy as np
import aerosandbox as asb
from geometry import *
class BEMAnalysis:

    def __init__(
        self,
        propeller: Propeller,
        rpm,
        velocity,
        rho=1.225,
        mu=1.81e-5,
    ):

        self.prop = propeller

        self.rpm = rpm
        self.velocity = velocity

        self.rho = rho
        self.mu = mu

        self.omega = rpm * 2 * np.pi / 60


    def run(self):

        thrust = 0.0
        torque = 0.0

        sections = []

        for i in range(self.prop.n_stations):
            
            r = self.prop.r[i]

            c = self.prop.chord[i]

            beta = np.deg2rad(self.prop.twist[i])   # degrees -> radians

            # -------------------------------------------------
            # No induction yet
            # -------------------------------------------------

            Vax = self.velocity

            Vtan = self.omega * r

            W = np.sqrt(
                Vax**2 +
                Vtan**2
            )

            phi = np.arctan2(
                Vax,
                Vtan,
            )

            alpha = beta - phi

            Re = (
                self.rho
                * W
                * c
                / self.mu
            )
            aero = self.prop.airfoil.get_aero_from_neuralfoil(
            alpha=np.rad2deg(alpha),
            Re=Re,
            mach=W/340,
)

            Cl = aero["CL"].item()
            Cd = aero["CD"].item()


            q = (
                0.5
                * self.rho
                * W**2
            )

            Lift = q * c * Cl

            Drag = q * c * Cd

            Fn = (
                Lift * np.cos(phi)
                - Drag * np.sin(phi)
            )

            Ft = (
                Lift * np.sin(phi)
                + Drag * np.cos(phi)
            )

            dT = (
                Fn
                * self.prop.n_blades
                * self.prop.dr
            )

            dQ = (
                Ft
                * r
                * self.prop.n_blades
                * self.prop.dr
            )

            thrust += dT

            torque += dQ

            sections.append(
                {
                    "r": r,
                    "twist_deg": self.prop.twist[i],
                    "phi_deg": np.rad2deg(phi),
                    "alpha_deg": np.rad2deg(alpha),
                    "Re": Re,
                    "Cl": Cl,
                    "Cd": Cd,
                    "Lift": Lift,
                    "Drag": Drag,
                    "dT": dT,
                    "dQ": dQ,
                }
            )

        power = torque * self.omega

        if self.velocity > 0:
            eta = thrust * self.velocity / power
        else:
            eta = 0

        return {
            "thrust": thrust,
            "torque": torque,
            "power": power,
            "efficiency": eta,
            "sections": sections,
        }