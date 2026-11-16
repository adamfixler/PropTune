import aerosandbox.numpy as np
import aerosandbox as asb
import casadi as ca
from geometry import *


class BEMAnalysis:

    def __init__(
        self,
        propeller: Propeller,
        rpm,
        velocity,
        rho=1.225,
        mu=1.81e-5,
        induction_guess=(0.02, 0.01),
        cl_offset=None,
        cd_offset=None,
        cd_round=1.17,
    ):
        self.prop = propeller
        self.rpm = rpm
        self.velocity = velocity
        self.rho = rho
        self.mu = mu
        self.omega = rpm * 2 * np.pi / 60
        self.induction_guess = induction_guess

        # Per-station additive corrections to NeuralFoil's Cl/Cd (e.g. fitted
        # to CFD by cfd_calibration.py). Zero by default = plain BEM. Applied
        # inside the induction solve too, so a/a' stay consistent with them.
        # Plain attributes: can be changed between run() calls without
        # rebuilding the solver.
        n = propeller.n_stations
        self.cl_offset = np.zeros(n) if cl_offset is None else cl_offset
        self.cd_offset = np.zeros(n) if cd_offset is None else cd_offset

        # Round sections (propeller.round_sections): no lift, drag
        # coefficient cd_round on the diameter (= that station's chord),
        # ~1.17 for a cylinder in crossflow at these Reynolds numbers.
        self.cd_round = cd_round

        # Built once; reused for every station and every design iterate.
        self._induction_solver = self._build_induction_solver()

    def _build_induction_solver(self):
        """CasADi rootfinder that solves for [a, a'] at a single blade
        station via Newton's method, given [r, chord, beta, Omega, dCl, dCd, w]
        (dCl/dCd: additive Cl/Cd corrections; w: 1 airfoil, 0 round section;
        see __init__).

        Solving via a rootfinder (rather than treating a/a' as free
        optimizer variables tied by an equality constraint) guarantees the
        specific, physically consistent root nearest the initial guess --
        equivalent to classical fixed-point BEM iteration, wrapped as a
        differentiable CasADi Function so it stays usable inside
        AeroSandbox's gradient-based optimizer.
        """
        B, R = self.prop.n_blades, self.prop.radius
        r_hub = self.prop.hub_radius
        rho, mu = self.rho, self.mu
        V = self.velocity
        airfoil = self.prop.airfoil

        z = ca.MX.sym("z", 2)            # [a, ap]
        a_s, ap_s = z[0], z[1]
        p = ca.MX.sym("p", 7)            # [r, chord, beta(rad), Omega, dCl, dCd, w]
        r_s, c_s, beta_s, Omega, dcl_s, dcd_s, w_s = p[0], p[1], p[2], p[3], p[4], p[5], p[6]

        Vax = V * (1 + a_s)
        Vtan = Omega * r_s * (1 - ap_s)
        phi = np.arctan2(Vax, Vtan)
        W = np.sqrt(Vax ** 2 + Vtan ** 2)
        alpha = beta_s - phi
        Re = rho * W * c_s / mu

        aero = airfoil.get_aero_from_neuralfoil(
            alpha=np.degrees(alpha), Re=Re, mach=W / 340,
        )
        # w = 1: airfoil; w = 0: round section.
        Cl = w_s * (aero["CL"] + dcl_s)
        Cd = w_s * (aero["CD"] + dcd_s) + (1 - w_s) * self.cd_round

        f_tip = (B / 2) * (R - r_s) / (r_s * np.sin(phi))
        f_tip = np.maximum(f_tip, 1e-6)
        F_tip = (2 / np.pi) * np.arccos(np.exp(-f_tip))

        # Prandtl hub-loss factor -- same derivation as tip loss, mirrored
        # about the root cutout. Reduces induction near the root the same
        # way the tip-loss factor does near the tip.
        f_hub = (B / 2) * (r_s - r_hub) / (r_s * np.sin(phi))
        f_hub = np.maximum(f_hub, 1e-6)  # singularity guard, mirrored from f_tip
        F_hub = (2 / np.pi) * np.arccos(np.exp(-f_hub))

        # Combined multiplicatively per standard Prandtl practice: each
        # factor -> 1 away from its own edge and -> 0 at it.
        F = F_tip * F_hub

        sigma = B * c_s / (2 * np.pi * r_s)
        Cn = Cl * np.cos(phi) - Cd * np.sin(phi)
        Ct = Cl * np.sin(phi) + Cd * np.cos(phi)

        resid_a = a_s * 4 * F * np.sin(phi) ** 2 - (1 + a_s) * sigma * Cn
        resid_ap = ap_s * 4 * F * np.sin(phi) * np.cos(phi) - (1 - ap_s) * sigma * Ct
        g_expr = ca.vertcat(resid_a, resid_ap)

        g_func = ca.Function("g", [z, p], [g_expr], ["z", "p"], ["g"])
        return ca.rootfinder("induction_solver", "newton", g_func)

    def run(self):
        thrust = 0.0
        torque = 0.0
        sections = []

        B = self.prop.n_blades

        for i in range(self.prop.n_stations):

            r = self.prop.r[i]
            c = self.prop.chord[i]
            beta = np.radians(self.prop.twist[i])  # degrees -> radians

            dcl, dcd = self.cl_offset[i], self.cd_offset[i]
            w = 0.0 if self.prop.round_sections[i] else 1.0

            z_sol = self._induction_solver(
                list(self.induction_guess), ca.vertcat(r, c, beta, self.omega, dcl, dcd, w)
            )
            a_i, ap_i = z_sol[0], z_sol[1]

            Vax = self.velocity * (1 + a_i)
            Vtan = self.omega * r * (1 - ap_i)
            phi = np.arctan2(Vax, Vtan)
            W = np.sqrt(Vax ** 2 + Vtan ** 2)
            alpha = beta - phi

            Re = self.rho * W * c / self.mu
            aero = self.prop.airfoil.get_aero_from_neuralfoil(
                alpha=np.degrees(alpha),
                Re=Re,
                mach=W / 340,
            )
            Cl = w * (aero["CL"] + dcl)
            Cd = w * (aero["CD"] + dcd) + (1 - w) * self.cd_round
            analysis_confidence = aero["analysis_confidence"]

            q = 0.5 * self.rho * W ** 2
            Lift = q * c * Cl
            Drag = q * c * Cd

            Fn = Lift * np.cos(phi) - Drag * np.sin(phi)
            Ft = Lift * np.sin(phi) + Drag * np.cos(phi)

            dT = Fn * B * self.prop.dr[i]
            dQ = Ft * r * B * self.prop.dr[i]

            thrust += dT
            torque += dQ

            sections.append(
                {
                    "r": r,
                    "twist_deg": self.prop.twist[i],
                    "phi_deg": np.degrees(phi),
                    "alpha_deg": np.degrees(alpha),
                    "Re": Re,
                    "Cl": Cl,
                    "Cd": Cd,
                    "analysis_confidence": analysis_confidence,
                    "a": a_i,
                    "ap": ap_i,
                    "Lift": Lift,
                    "Drag": Drag,
                    "dT": dT,
                    "dQ": dQ,
                }
            )

        power = torque * self.omega

        if self.velocity > 0:
            # Guard against near-zero/negative power (can occur at early
            # optimizer iterates) so eta doesn't blow up to inf/NaN.
            # np.maximum keeps this differentiable for CasADi.
            power_safe = np.maximum(power, 1e-6)
            eta = thrust * self.velocity / power_safe
        else:
            eta = 0

        return {
            "thrust": thrust,
            "torque": torque,
            "power": power,
            "efficiency": eta,
            "sections": sections,
        }
