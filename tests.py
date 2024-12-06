from config import R_BAR as R
from math import sqrt

ERR= 0.001

def checkPREOS(preos):
    dalpha_dT = -1 * preos.get_k() * sqrt(preos.T / preos.chemical.Tc) / preos.chemical.Tc
    da_dT = 0.45724 * (R**2) * (preos.chemical.Tc ** 2) * dalpha_dT / preos.chemical.Pc
    return abs(da_dT - preos.get_dadT()) < ERR

def checkMix(mix):
    a_mix = mix.eos[0].get_a() * (mix.inputs.x[0]**2) + 2 * mix.inputs.x[0] * mix.inputs.x[1] * \
        sqrt(mix.eos[0].get_a() * mix.eos[1].get_a()) * (1-mix.k) + \
        (mix.inputs.x[1] ** 2) * mix.eos[1].get_a()
    return abs(a_mix - mix.get_a_mix()) < ERR