from enum import Enum
from sympy import Integer as Int 
import sympy as sp
from src.utils import *

class Terminal:
    def __init__(self, name : str, unit : Int, weight : float = 1.):
        self.name = name
        self.unit = unit
        self.weight = weight
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"Terminal({self.name})"
    
    def __eq__(self, other):
        return self.name == other.name

class Operator:
    def __init__(self, name : str, arity : int, function : callable, unit_rule : callable, weight : float = 1.):
        self.name = name
        self.arity = arity
        self.function = function
        self.unit_rule = unit_rule
        self.weight = weight

    def __str__(self):
        return self.name
    
    def __call__(self, *args):
        return self.function(*args)
    
    def __repr__(self):
        return f"Operator({self.name}, arity={self.arity})"
    
    def __eq__(self, other):
        return self.name == other.name

# all terminals
OPEN = Terminal('open', Int(3), 0.5)
CLOSE = Terminal('close', Int(3), 0.5)
HIGH = Terminal('high', Int(3), 0.5)
LOW = Terminal('low', Int(3), 0.5)
VWAP = Terminal('vwap', Int(3), 0.5)
VOLUME = Terminal('volume', Int(2), 1.2)
ADV = Terminal('adv20', Int(2), 1.2)
RET = Terminal('returns', Int(1), 1) 
MOM = Terminal('ts_mean(returns, 63)', Int(1), 1)
BETA = Terminal('beta_last_60_days_spy', Int(1), 1)
ONE = Terminal('1', Int(1), 0)
ZERO = Terminal('0', Int(1), 0)
    
# all operators
ADD = Operator('add', 2, lambda x, y: f'add({x},{y})', lambda x, y : help_check_same(x, y))
SUB = Operator('sub', 2, lambda x, y: f'subtract({x},{y})', lambda x, y: help_check_same(x, y))   
MUL = Operator('mul', 2, lambda x, y: f'multiply({x},{y})', lambda x, y: x * y, weight=2/3)  # better not to use MUL on terminal nodes
DIV = Operator('div', 2, lambda x, y: f'divide({x},{y})', lambda x, y: x * y, weight=2/3)
MAX = Operator('max', 2, lambda x, y: f'max({x},{y})', lambda x, y : help_check_same(x, y))
MIN = Operator('min', 2, lambda x, y: f'min({x},{y})', lambda x, y : help_check_same(x, y))
REV = Operator('rev', 1, lambda x: f'reverse({x})', lambda x: x, weight=0.5)
SQRT = Operator('sqrt', 1, lambda x: f'sqrt({x})', lambda x: sp.sqrt(x), weight=0.5)
INV = Operator('inv', 1, lambda x: f'inverse({x})', lambda x: 1/x, weight=0.5)
SQUARE = Operator('square', 1, lambda x: f'power({x}, 2)',lambda x: x*x, weight=0.5)
ABS = Operator('abs', 1, lambda x: f'abs({x})', lambda x: x, weight=0.5)
SIGN = Operator('sign', 1, lambda x: f'sign({x})', lambda _: 1, weight=2/3)
CORR_21 = Operator('corr21', 2, lambda x, y: f'ts_corr({x},{y},21)', lambda _, _unused: 1, weight=1.2)
CORR_63 = Operator('corr63', 2, lambda x, y: f'ts_corr({x},{y},63)', lambda _, _unused: 1, weight=1.2)
TSZ_21 = Operator('tsz21', 1, lambda x: f'ts_zscore({x},21)', lambda _: 1, weight=1.2)
TSZ_63 = Operator('tsz63', 1, lambda x: f'ts_zscore({x},63)', lambda _: 1, weight=1.2)
REG_RESD_63 = Operator('reg_residuals63', 2, lambda x, y: f'ts_regression({x},{y},63,lag=0,rettype=0)', 
                        lambda _, y: y, weight=1.5)
REG_COEF_63 = Operator('reg_coefficient63', 2, lambda x, y: f'ts_regression({x},{y},63,lag=0,rettype=2)', 
                        lambda _, _unused: 1, weight=1.5)
RANK = Operator('rank', 1, lambda x: f'rank({x})', lambda _: 1, weight=1.5) 

ARITIES = {
    2 : [ADD, SUB, MUL, DIV, MAX, MIN, CORR_21, CORR_63, REG_RESD_63, REG_COEF_63],
    1 : [SQRT, INV, SQUARE, ABS, SIGN, RANK]
}

OPERATORS = [ADD, SUB, MUL, DIV, MAX, MIN, CORR_21, CORR_63, REG_RESD_63, REG_COEF_63, SQRT, INV, SQUARE, ABS, SIGN, RANK]
TERMINALS = [OPEN, CLOSE, HIGH, LOW, VWAP, VOLUME, ADV, RET, MOM, BETA, ONE, ZERO]
    
        
      