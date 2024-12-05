from enum import Enum
from dataclasses import dataclass
from typing import List

R=8.314
error=0.000001
n=1000
MAT_PROPS = ['x', 'Tm', 'delta_H_fus', 'gamma']
DEFAULT_COMP = [0.5, 0.5]

class Region(Enum):
    LEFT = 1
    RIGHT = 2
    ABOVE = 3

class Phase(Enum):
    SOLID = 'solid'
    LIQUID = 'liquid'
    SLE = 'sle'

class Code(Enum):
    IN = "Internal"
    USER = "User"
    EUT = "Eutectic"
    SOL = "Solution"
    LIQ = 'Composition'

class InputError(Exception):
    """Custom exception class for handling input validation errors"""
    def __init__(self, message: str, field: str = None, value: any = None, code: str = None):
        """
        Initialize the InputError
        
        Args:
            message (str): Error description
            field (str, optional): Name of the field that caused the error
            value (any, optional): Invalid value that caused the error
            code (str, optional): Error code for programmatic error handling
        """
        self.message = message
        self.field = field
        self.value = value
        self.code = code
        
        msg = message
        if code:
            msg = f"[Error Code: {code.value}] {message}"
        
        super().__init__(msg)
    def to_dict(self) -> dict:
        """Convert error to a dictionary format"""
        return {
            'message': self.message,
            'field': self.field,
            'value': self.value,
            'code': self.code
        }

class SolvingError(Exception):
    """Custom exception class for handling internal solver errors"""
    def __init__(self, message: str, field: str = None, value: any = None, code: str = None):
        """
        Initialize the SolvingError
        
        Args:
            message (str): Error description
            field (str, optional): Name of the field that caused the error
            value (any, optional): Invalid value that caused the error
            code (str, optional): Error code for programmatic error handling
        """
        self.message = message
        self.field = field
        self.value = value
        self.code = code
        
        msg = message
        if code:
            msg = f"[Error Code: {code.value}] {message}"
        
        super().__init__(msg)
    def to_dict(self) -> dict:
        """Convert error to a dictionary format"""
        return {
            'message': self.message,
            'field': self.field,
            'value': self.value,
            'code': self.code
        }

@dataclass
class Response:
    phase: Phase
    boundary: List[float]
    xe: float
    Te: float
    xS: List[float]
    xL: List[float]
    T: float
    liquid_frac: float
    solid_frac: float
    P: float

    def __post_init__(self):
        if not isinstance(self.phase, Phase):
            raise SolvingError('Found invalid phase.', code=Code.SOL)
        if len(self.boundary) != 2:
            raise SolvingError('SLE boundary is more than 2 dimensions.', code=Code.EUT)
        elif len(self.boundary[0]) != len(self.boundary[1]):
            raise SolvingError('SLE boundary points are mismatched.', code=Code.EUT)
        if self.xe < 0 or self.xe > 1:
            raise SolvingError('Invalid eutectic composition.', code=Code.EUT)
        if self.Te < 0:
            raise SolvingError('Invalid eutectic temperature.', code=Code.EUT)
        if len(self.xS) != 2:
            raise SolvingError('Expected binary solid composition.', code=Code.SOL)
        elif abs(sum(self.xS) - 1) > error:
            raise SolvingError('Solid composition must add up to 1.', code=Code.SOL)
        if len(self.xL) != 2:
            raise SolvingError('Expected binary liquid composition.', code=Code.SOL)
        elif abs(sum(self.xL) - 1) > error:
            raise SolvingError('Liquid composition must add up to 1.', code=Code.SOL)
        if self.T < 0:
            raise SolvingError('Invalid equilibrium temperature.', code=Code.LIQ)
        if self.liquid_frac < 0 or self.liquid_frac > 1:
            raise SolvingError('Invalid liquid fraction. Expected [0,1].', code=Code.LIQ)
        if self.solid_frac < 0 or self.solid_frac > 1:
            raise SolvingError('Invalid solid fraction. Expected [0,1].', code=Code.LIQ)
        if abs(sum([self.liquid_frac, self.solid_frac]) - 1) > error:
            raise SolvingError('Phase fractions must add up to 1.', code=Code.LIQ)
        if self.P < 0:
            raise SolvingError('Pressure should be greater than 0.', code=Code.IN)
            

@dataclass
class NRTLParams:
    x: List[float]
    T: float
    P: float
    delta_g12: float
    delta_g21: float
    alpha: float
    Tm: List[float]
    delta_H_fus: List[float]

    def __post_init__(self):
        if any([field is None for field in [self.x, self.T, self.P, self.delta_g12, self.delta_g21, self.alpha, self.Tm, self.delta_H_fus]]):
            raise InputError("All parameters must be filled.", code=Code.USER)
        if len(self.x) != 2:
            raise InputError("Expected binary composition.", code=Code.IN)
        if sum(self.x) != 1:
            raise InputError("Feed composition must sum to 1.", code=Code.USER)
        if any([xi > 1 for xi in self.x]):
            raise InputError("Species compositions must be within [0,1].", code=Code.USER)
        if self.T < 0:
            raise InputError("Temperature cannot be less than 0.", code=Code.USER)
        if self.P < 0:
            raise InputError("Pressure cannot be less than 0.", code=Code.USER)
        if len(self.Tm) != 2:
            raise InputError("Two melting temperatures expected.", code=Code.IN)
        if not all(T > 0 for T in self.Tm):
            raise InputError("Temperature cannot be less than 0.", code=Code.USER)
        if len(self.delta_H_fus) != 2:
            raise InputError("Two heat of fusions expected.", code=Code.IN)

class Species:
    def __init__(self, **kwargs):
        assert all([key in kwargs for key in MAT_PROPS])
        self.data = kwargs
    def __getitem__(self, key):
        return self.data[key]

"""Problem 3"""

R_BAR= 8.314 * (10 ** -5)

class Chemical:
    def __init__(self, **kwargs):
        self.name = kwargs['name']
        self.Tc = kwargs['Tc'] #K
        self.Pc = kwargs['Pc'] #bar
        self.antoine = {
            'A': kwargs['const_A'],
            'B': kwargs['const_B'],
            'C': kwargs['const_C']
        }

class State:
    def __init__(self, **kwargs):
        self.T = kwargs['T']
        self.P = kwargs['P']