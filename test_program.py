from src.program import Program
from src.function import ADD, MAX, INV, SQRT, OPEN, CLOSE, HIGH, LOW, VOLUME, RET, ADV, Operator, Terminal
from numpy.random import RandomState
import numpy as np

# Define a simple metric function
def simple_metric(expr):
    return 0.5  # Dummy value for testing

# First try with a manually created program in postfix notation
print("=== Testing manually created programs ===")
random_state = RandomState(42)

# In postfix notation: operands come before operators
# Try ADD(OPEN, CLOSE) -> [OPEN, CLOSE, ADD]
manual_program = [OPEN, CLOSE, ADD]
try:
    program = Program(
        max_depth=8,
        max_operators=15,
        metric=simple_metric,
        random_state=random_state,
        parimony_coefficient=0.1,
        p_point_replace=0.2,
        program=manual_program
    )
    print("OPEN CLOSE ADD validated successfully!")
    print(f"Program length: {program.length()}")
    print(f"Program depth: {program.depth()}")
    print(f"Program string representation: {program}")
except Exception as e:
    print(f"Error with OPEN CLOSE ADD: {e}")

# Try another operation: MAX(VOLUME, ADV) -> [VOLUME, ADV, MAX]
manual_program = [VOLUME, ADV, MAX]
try:
    program = Program(
        max_depth=8,
        max_operators=15,
        metric=simple_metric,
        random_state=random_state,
        parimony_coefficient=0.1,
        p_point_replace=0.2,
        program=manual_program
    )
    print("VOLUME ADV MAX validated successfully!")
    print(f"Program length: {program.length()}")
    print(f"Program depth: {program.depth()}")
    print(f"Program string representation: {program}")
except Exception as e:
    print(f"Error with VOLUME ADV MAX: {e}")

# Now try with automatic program generation
print("\n=== Testing automatic program generation ===")
for seed in range(10):
    print(f"\n=== Test with seed {seed} ===")
    random_state = RandomState(seed)
    
    # Try to initialize a Program with larger max_depth and max_operators
    try:
        program = Program(
            max_depth=10,
            max_operators=20,
            metric=simple_metric,
            random_state=random_state,
            parimony_coefficient=0.05,  # Lower penalty for program size
            p_point_replace=0.2
        )
        print("Program initialized successfully!")
        print(f"Program length: {program.length()}")
        print(f"Program depth: {program.depth()}")
        print(f"Program string representation: {program}")
    except Exception as e:
        print(f"Error initializing Program: {e}") 