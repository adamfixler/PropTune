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


def func(self, phi, v_inf, omega):
    """
    Residual function used in root-finding functions to find the inflow angle for the current section.

    .. math::
        \\frac{\\sin\\phi}{1+Ca} - \\frac{V_\\infty\\cos\\phi}{\\Omega R (1 - Ca\')} = 0\\\\

    :param float phi: Estimated inflow angle
    :param float v_inf: Axial inflow velocity
    :param float omega: Tangential rotational velocity
    :return: Residual
    :rtype: float
    """
    # Function to solve for a single blade element
    C = self.C

    a, ap = self.induction_factors(phi)
    
    resid = np.sin(phi)/(1 + C*a) - v_inf*np.cos(phi)/(omega*self.radius*(1 - C*ap))
    
    self.a = a
    self.ap = ap
    
    return resid