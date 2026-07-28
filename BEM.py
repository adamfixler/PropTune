import aerosandbox as asb
import aerosandbox.numpy as np


class BEMAnalysis:

    def __init__(
        self,
        propeller,
        rpm,
        velocity,
        rho=1.225,
        mu=1.81e-5,
    ):
        self.prop = propeller

        self.rpm = rpm
        self.omega = rpm * 2 * np.pi / 60

        self.V = velocity
        self.rho = rho
        self.mu = mu

    def run(self):

        r = self.prop.r
        c = self.prop.chord
        beta = self.prop.twist

        dr = np.diff(r)

        dr = np.concatenate([
            dr,
            np.array([dr[-1]])
        ])

        thrust = 0
        torque = 0

        sectional = []

        for i in range(len(r)):

            Ri = r[i]

            chord = c[i]

            twist = beta[i]

            # -------------------------------------
            # No induction (initial implementation)
            # -------------------------------------

            Vax = self.V

            Vtan = self.omega * Ri

            W = np.sqrt(
                Vax ** 2 +
                Vtan ** 2
            )

            phi = np.arctan(
                Vax / Vtan
            )

            alpha = twist - phi

            Re = (
                self.rho
                * W
                * chord
                / self.mu
            )

            Cl = self.prop.airfoil.CL_function(
                alpha,
                Re
            )

            Cd = self.prop.airfoil.CD_function(
                alpha,
                Re
            )

            Lift = (
                0.5
                * self.rho
                * W ** 2
                * chord
                * Cl
            )

            Drag = (
                0.5
                * self.rho
                * W ** 2
                * chord
                * Cd
            )

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
                * dr[i]
            )

            dQ = (
                Ft
                * Ri
                * self.prop.n_blades
                * dr[i]
            )

            thrust += dT
            torque += dQ

            sectional.append(
                dict(
                    r=Ri,
                    alpha=alpha,
                    Re=Re,
                    Cl=Cl,
                    Cd=Cd,
                    dT=dT,
                    dQ=dQ,
                )
            )

        power = torque * self.omega

        eta = thrust * self.V / power

        return dict(
            thrust=thrust,
            torque=torque,
            power=power,
            efficiency=eta,
            sections=sectional,
        )