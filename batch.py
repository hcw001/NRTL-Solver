from config import Chemical, R_BAR as R, V, kij, Response, State, BatchOutput, NRTLParams
from tests import checkPREOS, checkMix
from math import sqrt, log as ln, exp, log10
from scipy.optimize import fsolve

Benzaldehyde = Chemical(
    name='benzaldehyde',
    Tc=690,
    Pc=46.5,
    const_A=5.21496,
    const_B=2337.539,
    const_C=-5.103
)

Benzene = Chemical(
    name='benzene',
    Tc=562,
    Pc=48.9,
    const_A=4.72583,
    const_B=1660.652,
    const_C=-1.461
)

class PREOS:
    def __init__(self, chemical, T, P):
        self.chemical = chemical
        self.T = T
        self.P = P
    def get_Psat(self, T=None):
        """Antoine Equation"""
        if T is None: T = self.T
        A = self.chemical.antoine['A']
        B = self.chemical.antoine['B']
        C = self.chemical.antoine['C']
        return 10 ** (A - B / (T + C))
    def get_omega(self):
        """Accentric Factor"""
        return -1 - log10(self.get_Psat(0.7 * self.chemical.Tc) / self.chemical.Pc)
    def get_k(self):
        """(6.7-4)"""
        return 0.37464 + 1.54226*self.get_omega() - 0.26992*self.get_omega()**2
    def get_alpha(self):
        """(6.7-3)"""
        return (1 + self.get_k()*(1 - sqrt(self.T/self.chemical.Tc)))**2
    def get_a(self):
        """(6.7-1)"""
        return 0.45724 * (R**2) * (self.chemical.Tc**2) * self.get_alpha() / self.chemical.Pc
    def get_b(self):
        """(6.7-2)"""
        return 0.07780 * R * self.chemical.Tc / self.chemical.Pc
    def get_V(self):
        """(6.4-2): Complete generalized Peng-Robinson EOS"""
        obj = lambda V: R * self.T / (V - self.get_b()) - self.get_a() / (V * (V + self.get_b()) + self.get_b() * (V - self.get_b())) - self.P
        return fsolve(obj, 0.01)
    def get_Z(self):
        """(6.6-7)"""
        return self.P * self.get_V() / (R * self.T)
    def get_B(self):
        return self.P * self.get_b() / (R * self.T)
    def get_dadT(self):
        return -0.45724*(R**2)*(self.chemical.Tc**2) * self.get_k() * sqrt(self.get_alpha() / (self.T * self.chemical.Tc)) / self.chemical.Pc
    def get_H(self):
        """(6.4-29)"""
        return R * self.T * (self.get_Z() - 1) + \
            ((self.T * self.get_dadT() - self.get_a()) / (2 * self.get_b() * sqrt(2))) * \
            ln((self.get_Z() + (1 + sqrt(2))*self.get_B()) / (self.get_Z() + (1-sqrt(2)) * self.get_B()))

class Mix:
    def __init__(self, **kwargs):
        self.k = kwargs['k']
        self.eos = [kwargs['preos1'], kwargs['preos2']]
        self.nrtl = kwargs['nrtl']
        self.state = kwargs['state']
        self.inputs = kwargs['inputs']
        #Only handle binary mixtures
        assert len(self.inputs.x) == 2
        assert len(self.eos) == 2
    def get_aij(self):
        """(9.4-9): Equation-of-state combining rule"""
        return sqrt(self.eos[0].get_a() * self.eos[1].get_a())*(1-self.k)
    def get_a_mix(self):
        """(9.4-8): Equation-of-state mixing rule"""
        a_mix = 0
        aij = self.get_aij()
        for xi in self.inputs.x:
            for xj in self.inputs.x:
                a_mix += xi * xj * aij
        return a_mix
    def get_b_mix(self):
        """(9.4-8): Equation-of-state mixing rule"""
        b_mix = 0
        for i in range(2):
            b_mix += self.inputs.x[i] * self.eos[i].get_b()
        return b_mix
    def get_A_mix(self):
        return self.get_a_mix() * self.state.P / ((R**2)*(self.state.T**2))
    def get_B_mix(self):
        return self.get_b_mix() * self.state.P / (R * self.state.T)  
    def get_damixdT(self):
        return (self.inputs.x[0]**2)*self.eos[0].get_dadT() + 2*self.inputs.x[0]*self.inputs.x[1]*(1-self.k) * \
            (self.eos[1].get_a() * self.eos[0].get_dadT() + self.eos[0].get_a() * self.eos[1].get_dadT()) / \
            2*sqrt(self.eos[0].get_a()*self.eos[1].get_a()) + (self.inputs.x[1]**2)*self.eos[1].get_dadT()
    def get_Z(self):
        A_mix = self.get_A_mix()
        B_mix = self.get_B_mix()
        obj = lambda Z: (Z**3) + (Z**2) * (B_mix - 1) + Z*(A_mix - 2*B_mix - 3*(B_mix**2)) + (B_mix**2) * (B_mix + 1)
        return fsolve(obj, 1)
    def get_H(self):
        Z = self.get_Z()
        return R*self.state.T*(Z-1) + self.state.T * ((self.get_damixdT() - self.get_a_mix())/(2*sqrt(2) * self.get_b_mix())) * \
            ln((Z+(1+sqrt(2))*self.get_B_mix())/(Z+(1-sqrt(2))*self.get_B_mix()))
    def get_fs(self):
        B_mix = self.get_B_mix()
        Z = self.get_Z()
        f = []
        for i in range(2):
            lnphi = self.eos[i].get_b() * (self.state.P / (R*self.state.T)) * (Z-1)/(B_mix) - (self.get_A_mix() / (B_mix * sqrt(8))) * \
                ((2*self.get_aij()/self.get_a_mix()) - self.eos[i].get_b() / self.get_b_mix()) * \
                ln((Z+(1+sqrt(2))*B_mix)/(Z+(1-sqrt(2))*B_mix))
            f.append(exp(lnphi)*self.state.P)
        return f
    def get_delta_H_mix(self):
        """Definition of the enthalpy change on mixing"""
        return self.get_H() - self.inputs.x[0] * self.eos[0].get_H() - self.inputs.x[1] * self.eos[1].get_H()
    def get_n(self, V):
        Vmix = self.get_Z() * R * self.state.T / self.state.P
        return 5 / Vmix


def solve(inputs: NRTLParams, nrtl_state: Response):
    T = nrtl_state.T
    P = nrtl_state.P
    preos1 = PREOS(Benzaldehyde, T, P)
    assert checkPREOS(preos1)
    preos2 = PREOS(Benzene, T, P)
    assert checkPREOS(preos2)
    initial_state = State(T=T, P=P)
    mix = Mix(
        k=kij,
        preos1=preos1,
        preos2=preos2,
        nrtl=nrtl_state,
        state=initial_state,
        inputs=inputs
    )
    assert checkMix(mix)
    #Solve for Heat Requirement
    n = mix.get_n(V)
    Hfus = sum([nrtl_state.xS[i] * inputs.delta_H_fus[i] * n * nrtl_state.solid_frac for i in range(2)])
    Q = mix.get_delta_H_mix() * n - Hfus
    #Solve for Pressure
    Pf = nrtl_state.xL[0] * preos1.get_Psat(T) + nrtl_state.xL[1] * preos2.get_Psat(T)
    y1 = nrtl_state.xL[0] * preos1.get_Psat(T) / Pf
    return BatchOutput(
        Tf= T,
        Pf= Pf,
        Q= Q,
        xL= nrtl_state.xL,
        xS= nrtl_state.xS,
        solid_frac= nrtl_state.solid_frac,
        liquid_frac= nrtl_state.liquid_frac,
        y=[y1, 1-y1]
    )
