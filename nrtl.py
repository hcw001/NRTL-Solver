import math
import numpy as np
from scipy.optimize import fsolve
from config import Region, Phase, Species, NRTLParams, Response, \
    R, error, n, MAT_PROPS, DEFAULT_COMP

#Helper Functions
def sq(n): return n**2
def ln(n): return np.log(np.maximum(n,1e-10))

def activity_coeffs(x, T, delta_g12, delta_g21, alpha):
    """
    Find activity coefficient using NRTL model.
    Based on three-parameter nonrandom two-liquid equation.
    Inputs:
        x: float[2] - binary compositions
        T: float - temperature [Kelvin] (>0)
        delta_g12: float - ...
        delta_g21: float - ...
        alpha: float - non-randomness parameter
    Returns: activity coefficients <gamma>1 & <gamma>2
    """
    x1, x2 = x
    tau12 = delta_g12 / (R * T)
    tau21 = delta_g21 / (R * T)
    G12 = math.exp(-1 * alpha * tau12)
    G21 = math.exp(-1 * alpha * tau21)
    gamma1 = math.exp(sq(x2)*(tau21*sq(G21/(x1+x2*G21))+(tau12*G12)/sq(x2+x1*G12)))
    gamma2 = math.exp(sq(x1)*(tau12*sq(G12/(x2+x1*G12))+(tau21*G21)/sq(x1+x2*G21)))
    return gamma1, gamma2

def Tfsolve0(x, species):
    """Assuming Tm & Tf are sufficiently close"""
    x=1-x
    Tm = species['Tm']
    gamma = species['gamma']
    delta_H_fus = species['delta_H_fus']
    #return lambda Tf: Tm - Tf - R * sq(Tm) * ln(x*gamma) / delta_H_fus
    return lambda Tf: ln(x*gamma) + delta_H_fus*(1/Tf-1/Tm)/R

def find_eutectic_point(species1, species2):
    """Solving For Eutectic Point"""
    xs = [x/n for x in range(1,n)]
    y1 = np.array([fsolve(Tfsolve0(x, species1), species1['Tm']) for x in xs])
    y2 = np.array([fsolve(Tfsolve0(x, species2), species2['Tm']) for x in list(reversed(xs))])
    #Find intersection of curves
    diff = np.signbit(y1 - y2)
    sign_change = np.bitwise_xor(diff[:-1], diff[1:])
    ie = np.where(sign_change)[0][0]
    #Eutectic Composition Species 1
    xe = xs[ie]
    #Eutectic Temperature
    Te = (y1[ie]+y2[ie])/2
    #Boundary
    boundary = np.ndarray((2, len(xs)))
    boundary[0,:] = xs
    boundary[1, :ie] = y1[:ie,0]
    boundary[1, ie:] = y2[ie:,0]
    return xe, Te, boundary

def within_boundary(boundary, xe, x1, T, n):
    """Check is provided parameters are within SLE boudary"""
    x1 = round(x1, int(math.log10(n)))
    i = np.where(boundary[0,:] == x1)[0]
    Tf = boundary[1,i]
    if T < Tf:
        if x1 < xe: return Region.LEFT
        else: return Region.RIGHT
    return Region.ABOVE
    
#Tried using NRTL parameters to solve for equilibrium
# def liquid_comp(species1, species2, T0, delta_g12, delta_g21, alpha):
#     def ln_gamma(x,T):
#         tau12 = delta_g12 / (R * T)
#         tau21 = delta_g21 / (R * T)
#         G12 = math.exp(-1 * alpha * tau12)
#         G21 = math.exp(-1 * alpha * tau21)
#         return sq(1-x)*(tau21*sq(G21/(x+(1-x)*G21))+(tau12*G12)/sq((1-x)+x*G12))
    
#     def eqns(vars):
#         x, T = vars
#         return [
#             ln(x) + ln_gamma(x,T) + species1['delta_H_fus'] * (1 - T/species1['Tm']) / (R * T),
#             ln(1-x) + ln_gamma(1-x,T) + species2['delta_H_fus'] * (1 - T/species2['Tm']) / (R * T),
#         ]
#     x1, T = fsolve(eqns, [0.5, T0])
#     return x1, T 

def find_intersect(region, x1, T, boundary):
    """Helper function for calculating lever rule"""
    lever= np.ones(boundary.shape[1]) * T
    diff = np.signbit(boundary[1,:] - lever)
    sign_change = np.bitwise_xor(diff[:-1], diff[1:])
    ie = np.where(sign_change)[0][0]
    return boundary[0,ie]

def solve_eq(region, x1, T, boundary):
    """Utilize lever rule to solve equilibrium conditions """
    HK = x1 if region == Region.LEFT else 1-x1 #Amount of Liquid
    xL = find_intersect(region, x1, T, boundary)
    HL = abs(x1-xL) #Amount of Solid
    return {
        'liquid_frac': HK / (HK + HL),
        'solid_frac': HL / (HK + HL),
        'xL': xL
    }

def NRTL(params: NRTLParams):
    global n
    T = params.T
    P = params.P
    gamma1, gamma2 = activity_coeffs(
        **{k: getattr(params, k) for k in ['x', 'T', 'delta_g12', 'delta_g21', 'alpha']}
    )
    species1 = Species(**{**{k: getattr(params,k, 'NAN')[0] for k in MAT_PROPS}, **{'gamma': gamma1}})
    species2 = Species(**{**{k: getattr(params,k, 'NAN')[1] for k in MAT_PROPS}, **{'gamma': gamma2}})
    xe, Te, boundary = find_eutectic_point(species1, species2)
    if T < Te:
        #Solid
        return Response(
            phase= Phase.SOLID,
            boundary= boundary,
            xe= xe,
            Te= Te,
            xS= params.x,
            xL= DEFAULT_COMP,
            T= T,
            P=P,
            liquid_frac=0,
            solid_frac=1,
        )
    else:
        region = within_boundary(boundary, xe, species1['x'], T, n)
        if region == Region.ABOVE:
            #Liquid
            return Response(
                phase= Phase.LIQUID,
                boundary= boundary,
                xe= xe,
                Te= Te,
                xS= DEFAULT_COMP,
                xL= params.x,
                T= T,
                P=P,
                liquid_frac=1,
                solid_frac=0
            )
        elif region == Region.LEFT:
            #SLE - Species 2 - Region 1
            #xL1, Teq = liquid_comp(species1, species2, T, params.delta_g12, params.delta_g21, params.alpha)
            eq = solve_eq(region, params.x[0], T, boundary)
            return Response(
                phase= Phase.SLE,
                boundary= boundary,
                xe= xe,
                Te= Te,
                xS= [0,1],
                xL= [eq['xL'], 1-eq['xL']],
                T= T,
                P=P,
                liquid_frac=eq['liquid_frac'],
                solid_frac=eq['solid_frac']
        )
        elif region == Region.RIGHT:
            #SLE - Species 1 - Region 2
            #xL1, Teq = liquid_comp(species1, species2, T, params.delta_g12, params.delta_g21, params.alpha)
            eq = solve_eq(region, params.x[0], T, boundary)
            return Response(
                phase= Phase.SLE,
                boundary= boundary,
                xe= xe,
                Te= Te,
                xS= [1,0],
                xL= [eq['xL'], 1-eq['xL']],
                T= T,
                P=P,
                liquid_frac=eq['liquid_frac'],
                solid_frac=eq['solid_frac']
            )
        else: raise ValueError("Unexpected Region.")