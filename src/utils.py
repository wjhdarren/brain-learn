from sympy import Integer as Int



def help_check_same(x : Int, y : Int) -> bool:
    if x == y:
        return x
    else:
        raise ValueError('This binary operator requires the same unit for both inputs')

