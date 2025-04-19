import requests
from sympy import Integer as Int
from src.brain import simulate
from functools import partial

def evaluate_fitness(s : requests.Session):
    return partial(simulate, s)


def help_check_same(x : Int, y : Int) -> bool:
    if x == y:
        return x
    else:
        raise ValueError('This binary operator requires the same unit for both inputs')

