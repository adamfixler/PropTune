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

            beta = np.radians(self.prop.twist[i])   # degrees -> radians

            # -------------------------------------------------
            # No induction yet
            # -------------------------------------------------
            a,ap = induction_factors(self,phi)

            Vax = self.velocity * (1 + a)

            Vtan = self.omega * r * (1 + ap)

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
            alpha=np.degrees(alpha),
            Re=Re,
            mach=W/340,
)

            Cl = aero["CL"]
            Cd = aero["CD"]


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



def prandtl(self,dr, r, phi):
    f = self.rotor.n_blades*dr/(2*r*(np.sin(phi)))
    if (-f > 500): # exp can overflow for very large numbers
        F = 1.0
    else:
        F = 2*np.acos(min(1.0, np.exp(-f)))/np.pi
        
    return F

def induction_factors(self, phi):
    """
    Calculation of axial and tangential induction factors,

    .. math::
        a = \\frac{1}{\\kappa - C} \\\\
        a\' = \\frac{1}{\\kappa\' + C} \\\\
        \\kappa = \\frac{4F\\sin^2{\\phi}}{\\sigma C_T} \\\\
        \\kappa\' = \\frac{4F\\sin{\\phi}\\cos{\\phi}}{\\sigma C_Q} \\\\
        
    :param float phi: Inflow angle
    :return: Axial and tangential induction factors
    :rtype: tuple
    """

    C = self.C
    
    F = self.tip_loss(phi)
    
    CT, CQ = self.airfoil_forces(phi)
    
    kappa = 4*F*sin(phi)**2/(self.sigma*CT)
    kappap = 4*F*sin(phi)*cos(phi)/(self.sigma*CQ)

    a = 1.0/(kappa - C)
    ap = 1.0/(kappap + C)
    
    return a, ap

def brute_solve(self, sec, v, omega, n=3600):
        """ 
        Solve by a simple brute force procedure, iterating through all
        possible angles and selecting the one with lowest residual.

        :param Section sec: Section to solve for
        :param float v: Axial inflow velocity
        :param float omega: Tangential rotational velocity
        :param int n: Number of angles to test for, optional
        :return: Inflow angle with lowest residual
        :rtype: float
        """
        resid = np.zeros(n)
        phis = np.linspace(-0.9*np.pi,0.9*np.pi,n)
        for i,phi in enumerate(phis):
            res = sec.func(phi, v, omega)
            if not np.isnan(res):
                resid[i] = res
            else:
                resid[i] = 1e30
        i = np.argmin(abs(resid))
        return phis[i]