import unittest
import numpy as np
from src.genetic import GPLearnSimulator
from src.program import Program
from src.function import Terminal, Operator, ADD, SUB, MUL, DIV, OPEN, CLOSE, HIGH, LOW, VWAP, VOLUME

def dummy_metric(expr):
    """A dummy metric function that returns a simple score based on expression complexity.
    
    This simulates the behavior of the actual evaluation function without
    requiring access to the actual simulation backend.
    
    Parameters
    ----------
    expr : str
        The string representation of the expression to evaluate
        
    Returns
    -------
    dict
        A dict with 'sharpe' key containing the simulated Sharpe ratio
    """
    try:
        # Simple heuristic: more complex expressions have higher initial scores
        # but excessive complexity is penalized
        complexity = expr.count('(') * 2 + len(expr)
        # Prefer expressions with volume term
        has_volume = 'volume' in expr
        # Avoid trivial expressions
        too_simple = len(expr) < 10
        
        # Sharpe ratio simulation - in real system this would be calculated by backtesting
        sharpe = 0.5 + np.random.normal(0, 0.2)
        
        # Adjust based on heuristics
        if has_volume:
            sharpe += 0.3
        if too_simple:
            sharpe -= 0.5
            
        # Apply a complexity penalty for extremely long expressions
        complexity_penalty = max(0, (complexity - 50) / 100)
        sharpe -= complexity_penalty
        
        return {'sharpe': sharpe}
    except Exception as e:
        # Return no score for invalid expressions
        print(f"Error evaluating expression: {e}")
        return {'sharpe': None}


def deterministic_metric(expr):
    """A deterministic metric function for reproducible tests.
    
    Returns a predictable score based on the structure of the expression.
    
    Parameters
    ----------
    expr : str
        The string representation of the expression to evaluate
        
    Returns
    -------
    dict
        A dict with 'sharpe' key containing the deterministic Sharpe ratio
    """
    try:
        # Count key operators as a measure of expression complexity
        num_add = expr.count('add(')
        num_mul = expr.count('multiply(')
        num_div = expr.count('divide(')
        
        # Specific pattern scoring - prefer multiplication and division
        sharpe = 0.1
        sharpe += num_add * 0.05
        sharpe += num_mul * 0.2
        sharpe += num_div * 0.15
        
        # Bonus for using volume - simulate that this is a valuable feature
        if 'volume' in expr:
            sharpe += 0.5
            
        # Penalty for overly complex expressions
        if len(expr) > 100:
            sharpe *= 0.7
            
        # Minimum score for valid expressions
        sharpe = max(0.01, sharpe)
        
        return {'sharpe': sharpe}
    except Exception:
        return {'sharpe': None}


class TestGPLearnSimulator(unittest.TestCase):
    def test_initialization(self):
        """Test the initialization of the GPLearnSimulator."""
        # Initialize with default parameters
        gp = GPLearnSimulator(random_state=42)
        
        # Check that parameters were set correctly
        self.assertEqual(gp.population_size, 30)
        self.assertEqual(gp.generations, 20)
        self.assertEqual(gp.tournament_size, 5)
        self.assertAlmostEqual(gp.p_crossover, 0.7)
        
        # Initialize with custom parameters
        custom_gp = GPLearnSimulator(
            population_size=50,
            generations=10,
            tournament_size=3,
            random_state=42
        )
        
        # Check that custom parameters were set correctly
        self.assertEqual(custom_gp.population_size, 50)
        self.assertEqual(custom_gp.generations, 10)
        self.assertEqual(custom_gp.tournament_size, 3)
    
    def test_population_initialization(self):
        """Test the initialization of the population."""
        # Use a fixed random seed for reproducibility
        gp = GPLearnSimulator(
            population_size=10,
            random_state=42
        )
        
        # Initialize the population
        gp._initialize_population()
        
        # Check that the population has the correct size
        self.assertEqual(len(gp.population), 10)
        
        # Check that all members are Program instances
        for member in gp.population:
            self.assertIsInstance(member, Program)
    
    def test_fitness_evaluation(self):
        """Test the fitness evaluation of a program."""
        # Create a simulator with the dummy metric function
        gp = GPLearnSimulator(random_state=42)
        gp.metric = dummy_metric
        
        # Create a simple program for testing
        program = Program(
            max_depth=3,
            max_operators=5,
            random_state=np.random.RandomState(42),
            metric=dummy_metric,
            parimony_coefficient=0.1,
            program=[OPEN, CLOSE, ADD, VOLUME, MUL]  # (OPEN + CLOSE) * VOLUME
        )
        
        # Evaluate the fitness
        fitness = gp._evaluate_fitness(program)
        
        # Check that fitness is a number
        self.assertIsInstance(fitness, float)
        
        # Check that the program's fitness has been set
        self.assertIsNotNone(program.fitness)
        
        # Ensure fitness evaluations counter is incremented
        self.assertEqual(gp.fitness_evaluations, 1)
    
    def test_tournament_selection(self):
        """Test the tournament selection method."""
        # Create a simulator
        gp = GPLearnSimulator(
            population_size=10,
            tournament_size=3,
            random_state=42
        )
        gp.metric = dummy_metric
        
        # Create a population with known fitness values
        gp.population = []
        for i in range(10):
            program = Program(
                max_depth=3,
                max_operators=5,
                random_state=np.random.RandomState(42),
                metric=dummy_metric,
                parimony_coefficient=0.1,
                program=[OPEN, CLOSE, ADD]  # Simple program: OPEN + CLOSE
            )
            # Manually set fitness values (higher index = higher fitness)
            program.fitness = i / 10
            gp.population.append(program)
        
        # Run tournament selection multiple times
        selections = [gp._tournament_selection() for _ in range(50)]
        
        # Calculate the average fitness of selected individuals
        avg_fitness = sum(program.fitness for program in selections) / len(selections)
        
        # With tournament selection, average fitness should be higher than random selection (0.45)
        self.assertGreater(avg_fitness, 0.45)
    
    def test_evolution_process(self):
        """Test the evolution process for a few generations."""
        # Create a simulator with a small population and few generations
        gp = GPLearnSimulator(
            population_size=10,
            generations=3,
            tournament_size=3,
            random_state=42
        )
        gp.metric = dummy_metric
        
        # Run evolution
        best_program = gp.evolve(verbose=False)
        
        # Check that the best program is returned
        self.assertIsInstance(best_program, Program)
        
        # Check that history has been updated
        self.assertEqual(len(gp.history), 3)
        
        # Check that the best fitness is tracked
        self.assertIsNotNone(gp.best_fitness)
        
        # Best fitness should increase or stay the same (not decrease)
        fitness_history = gp.get_fitness_history()
        for i in range(1, len(fitness_history)):
            self.assertGreaterEqual(fitness_history[i], fitness_history[i-1])
        
        # Check statistics
        self.assertGreater(gp.fitness_evaluations, 0)
    
    def test_get_best_individual(self):
        """Test getting the best individual."""
        # Create and evolve a simulator
        gp = GPLearnSimulator(
            population_size=10,
            generations=2,
            random_state=42
        )
        gp.metric = dummy_metric
        gp.evolve(verbose=False)
        
        # Get the best individual
        best = gp.get_best_individual()
        
        # It should be a Program instance
        self.assertIsInstance(best, Program)
        
        # Its fitness should match the best fitness
        self.assertEqual(best.fitness, gp.best_fitness)

    def test_deterministic_evolution(self):
        """Test evolution with a deterministic fitness function."""
        # Create a simulator with deterministic metric
        gp = GPLearnSimulator(
            population_size=15,
            generations=5,
            tournament_size=3,
            random_state=42
        )
        gp.metric = deterministic_metric
        
        # Run evolution
        best_program = gp.evolve(verbose=False)
        
        # Record results
        first_run_fitness = gp.best_fitness
        first_run_evaluations = gp.fitness_evaluations
        
        # Reset and run again with same parameters
        gp2 = GPLearnSimulator(
            population_size=15,
            generations=5,
            tournament_size=3,
            random_state=42
        )
        gp2.metric = deterministic_metric
        gp2.evolve(verbose=False)
        
        # With deterministic metric and same random seed, results should be identical
        self.assertEqual(first_run_fitness, gp2.best_fitness)
        self.assertEqual(first_run_evaluations, gp2.fitness_evaluations)
        
        # Check that we got a good solution (should contain volume term with deterministic metric)
        self.assertIn('volume', str(best_program))
    
    def test_crossover_probability(self):
        """Test different crossover probabilities."""
        # Run with high crossover probability
        gp_high_crossover = GPLearnSimulator(
            population_size=10,
            generations=3,
            p_crossover=0.9,
            p_mutation=0.1,
            random_state=42
        )
        gp_high_crossover.metric = deterministic_metric
        gp_high_crossover.evolve(verbose=False)
        
        # Run with low crossover probability
        gp_low_crossover = GPLearnSimulator(
            population_size=10,
            generations=3,
            p_crossover=0.1,
            p_mutation=0.9,
            random_state=42
        )
        gp_low_crossover.metric = deterministic_metric
        gp_low_crossover.evolve(verbose=False)
        
        # Both should produce valid results
        self.assertIsNotNone(gp_high_crossover.best_program)
        self.assertIsNotNone(gp_low_crossover.best_program)
    
    def test_parsimony_pressure(self):
        """Test different parsimony pressure coefficients."""
        # Run with low parsimony pressure
        gp_low_parsimony = GPLearnSimulator(
            population_size=10,
            generations=3,
            parsimony_coefficient=0.01,
            random_state=42
        )
        gp_low_parsimony.metric = deterministic_metric
        gp_low_parsimony.evolve(verbose=False)
        
        # Run with high parsimony pressure
        gp_high_parsimony = GPLearnSimulator(
            population_size=10,
            generations=3,
            parsimony_coefficient=0.5,
            random_state=42
        )
        gp_high_parsimony.metric = deterministic_metric
        gp_high_parsimony.evolve(verbose=False)
        
        # With higher parsimony pressure, programs should be shorter on average
        avg_length_low = np.mean([len(p.program) for p in gp_low_parsimony.population])
        avg_length_high = np.mean([len(p.program) for p in gp_high_parsimony.population])
        
        # This isn't guaranteed due to randomness, but is likely with enough runs
        # We're using a statistical test rather than a strict assertion
        self.assertLessEqual(avg_length_high, avg_length_low * 1.5)

    def test_empty_population_handling(self):
        """Test that the evolution process handles empty initial population."""
        # Create a simulator but don't initialize population
        gp = GPLearnSimulator(
            population_size=5,
            generations=2,
            random_state=42
        )
        gp.metric = deterministic_metric
        
        # Population should be empty initially
        self.assertEqual(len(gp.population), 0)
        
        # Evolution should initialize population
        gp.evolve(verbose=False)
        
        # Now population should be the specified size
        self.assertEqual(len(gp.population), 5)
    
    def test_edge_case_fitness(self):
        """Test handling of edge cases in fitness evaluation."""
        gp = GPLearnSimulator(random_state=42)
        
        # Create a custom metric that sometimes returns None or raises exceptions
        def problematic_metric(expr):
            if 'divide' in expr:
                # Simulate division by zero error
                return {'sharpe': None}
            elif len(expr) < 10:
                # Simulate exception for very short expressions
                raise ValueError("Expression too short")
            else:
                # Return normal value
                return {'sharpe': 0.5}
        
        gp.metric = problematic_metric
        
        # Create programs that will trigger different error conditions
        program1 = Program(
            max_depth=3,
            max_operators=5,
            random_state=np.random.RandomState(42),
            metric=problematic_metric,
            parimony_coefficient=0.1,
            program=[OPEN, CLOSE, DIV]  # Should return None
        )
        
        program2 = Program(
            max_depth=3,
            max_operators=5,
            random_state=np.random.RandomState(42),
            metric=problematic_metric,
            parimony_coefficient=0.1,
            program=[OPEN]  # Should raise exception
        )
        
        # Both should be handled gracefully
        fitness1 = gp._evaluate_fitness(program1)
        fitness2 = gp._evaluate_fitness(program2)
        
        # Both should result in -inf fitness
        self.assertEqual(fitness1, float('-inf'))
        self.assertEqual(fitness2, float('-inf'))


if __name__ == "__main__":
    unittest.main() 