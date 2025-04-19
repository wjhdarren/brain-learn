import requests
import numpy as np
import time
import os
from src.utils import evaluate_fitness
from datetime import timedelta
from src.program import Program

# Maximum number of reconnection attempts
MAX_RECONNECTION_ATTEMPTS = 3

class GPLearnSimulator:
    def __init__(self, 
                 population_size=30,
                 generations=20,
                 tournament_size=5,
                 p_crossover=0.7,
                 p_mutation=0.1,
                 p_subtree_mutation=0.05,
                 p_hoist_mutation=0.05,
                 p_point_mutation=0.1,
                 max_depth=5,
                 max_operators=10,
                 session = None,
                 random_state=None,
                 parsimony_coefficient=0.15):
        """
        A Genetic Programming simulator optimized for simulation-based fitness evaluation.
        
        Parameters
        ----------
        population_size : int
            Size of the population
        generations : int
            Number of generations to evolve
        tournament_size : int
            Size of tournament for selection
        p_crossover : float
            Probability of crossover
        p_mutation : float
            Probability of mutation
        p_subtree_mutation : float
            Probability of subtree mutation
        p_hoist_mutation : float
            Probability of hoist mutation
        p_point_mutation : float
            Probability of point mutation
        max_depth : int
            Maximum tree depth
        max_operators : int
            Maximum number of operators
        random_state : int or RandomState, optional
            Random number generator
        metric : callable, optional
            Function to evaluate fitness
        parsimony_coefficient : float
            Coefficient for parsimony pressure (penalty for size)
        """
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.p_subtree_mutation = p_subtree_mutation / (p_subtree_mutation + p_hoist_mutation + p_point_mutation)
        self.p_hoist_mutation = p_hoist_mutation / (p_subtree_mutation + p_hoist_mutation + p_point_mutation)
        self.p_point_mutation = p_point_mutation / (p_subtree_mutation + p_hoist_mutation + p_point_mutation)
        self.max_depth = max_depth
        self.max_operators = max_operators
        self.session = session
        self.metric = evaluate_fitness(session)
        self.parsimony_coefficient = parsimony_coefficient
        
        # Setup random state
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state
            
        # Initialize tracking variables
        self.population = []
        self.history = []
        self.best_program = None
        self.best_fitness = float('-inf')
        self.generation = 0
        
        # Statistics
        self.fitness_evaluations = 0
        self.start_time = None
        
    def _initialize_population(self):
        """Initialize a population of random programs."""
        self.population = []
        for _ in range(self.population_size):
            program = Program(
                max_depth=self.max_depth,
                max_operators=self.max_operators,
                random_state=self.random_state,
                metric=self.metric,
                parimony_coefficient=self.parsimony_coefficient
            )
            self.population.append(program)
    
    def _recreate_session(self):
        """Recreate the session if authentication fails."""
        from dotenv import load_dotenv
        
        # Load credentials from .env file
        load_dotenv()
        username = os.getenv("USERNAME")
        password = os.getenv("PASSWORD")
        
        # Create a new session
        self.session = requests.Session()
        self.session.auth = (username, password)
        
        # Authenticate the session
        response = self.session.post('https://api.worldquantbrain.com/authentication')
        
        if response.status_code == 201:
            print("Session reconnected successfully.")
            # Update the metric function with the new session
            self.metric = evaluate_fitness(self.session)
            return True
        else:
            print(f"Failed to reconnect session. Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    def _evaluate_fitness(self, program):
        """Evaluate the fitness of a program."""
        if program.fitness is None:
            # Calculate raw fitness using the Program's raw_fitness method
            fast_expr = program.__str__()
            
            # Try multiple times in case of authentication failure
            for attempt in range(MAX_RECONNECTION_ATTEMPTS):
                try:
                    result = self.metric(fast_expr)
                    
                    # Check if the result contains an error message about authentication
                    if result is None:
                        # Try to recreate the session
                        if self._recreate_session():
                            continue  # Try again with the new session
                        else:
                            program.raw_fitness = float('-inf')
                            break
                    
                    sharpe = result.get('sharpe')
                    if sharpe is None:
                        program.raw_fitness = float('-inf')
                    else:
                        program.raw_fitness = sharpe
                    break  # Success, exit the retry loop
                    
                except Exception as e:
                    error_message = str(e)
                    if "401" in error_message or "authentication" in error_message.lower():
                        # Authentication error, attempt to reconnect
                        if self._recreate_session():
                            continue  # Try again with the new session
                    
                    program.raw_fitness = float('-inf')
                    if attempt == MAX_RECONNECTION_ATTEMPTS - 1:
                        print(f"Failed to evaluate fitness after {MAX_RECONNECTION_ATTEMPTS} attempts: {e}")
                
            # Calculate fitness with parsimony pressure
            program.fitness = program.raw_fitness - program.parimony_coefficient * len(program.program)
            self.fitness_evaluations += 1
            
        return program.fitness
    
    def _tournament_selection(self):
        """Select an individual using tournament selection."""
        indices = self.random_state.randint(0, len(self.population), self.tournament_size)
        tournament = [self.population[i] for i in indices]
        
        # Return the best individual in the tournament
        return max(tournament, key=lambda program: self._evaluate_fitness(program))
    
    def _update_best(self):
        """Update the best program seen so far."""
        current_best = max(self.population, key=lambda program: self._evaluate_fitness(program))
        current_best_fitness = self._evaluate_fitness(current_best)
        
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_program = current_best
    
    def evolve(self, verbose=True, log_interval=1):
        """
        Run the genetic programming evolution process.
        
        Parameters
        ----------
        verbose : bool
            Whether to print progress information
        log_interval : int
            Interval (in generations) at which to log progress
            
        Returns
        -------
        Program
            The best program found
        """
        if not self.population:
            self._initialize_population()
            
        self.start_time = time.time()
        
        for gen in range(self.generations):
            self.generation = gen
            
            # Create a new generation
            new_population = []
            
            # Elite selection - keep the best individual
            elite = max(self.population, key=lambda program: self._evaluate_fitness(program))
            new_population.append(elite)
            
            # Fill the rest of the population
            while len(new_population) < self.population_size:
                op_choice = self.random_state.uniform()
                
                if op_choice < self.p_crossover and len(new_population) < self.population_size - 1:
                    # Crossover
                    parent1 = self._tournament_selection()
                    parent2 = self._tournament_selection()
                    
                    # Ensure parents are different
                    attempts = 0
                    while parent1 is parent2 and attempts < 10:
                        parent2 = self._tournament_selection()
                        attempts += 1
                    
                    offspring1_program, _, _ = parent1.crossover(parent2.program, self.random_state)
                    offspring2_program, _, _ = parent2.crossover(parent1.program, self.random_state)
                    
                    offspring1 = Program(
                        max_depth=self.max_depth,
                        max_operators=self.max_operators,
                        random_state=self.random_state,
                        metric=self.metric,
                        parimony_coefficient=self.parsimony_coefficient,
                        program=offspring1_program
                    )
                    
                    offspring2 = Program(
                        max_depth=self.max_depth,
                        max_operators=self.max_operators,
                        random_state=self.random_state,
                        metric=self.metric,
                        parimony_coefficient=self.parsimony_coefficient,
                        program=offspring2_program
                    )
                    
                    new_population.append(offspring1)
                    if len(new_population) < self.population_size:
                        new_population.append(offspring2)
                else:
                    # Mutation
                    parent = self._tournament_selection()
                    mutation_op = self.random_state.uniform()
                    
                    if mutation_op < self.p_subtree_mutation:
                        # subtree_mutation returns (program, removed, donor_removed)
                        mutation_result = parent.subtree_mutation(self.random_state)
                        offspring_program = mutation_result[0]  # Just get the program
                    elif mutation_op < self.p_subtree_mutation + self.p_hoist_mutation:
                        # hoist_mutation returns (program, removed)
                        offspring_program, _ = parent.hoist_mutation(self.random_state)
                    else:
                        # point_mutation returns (program, mutated_indices)
                        offspring_program, _ = parent.point_mutation(self.random_state)
                    
                    offspring = Program(
                        max_depth=self.max_depth,
                        max_operators=self.max_operators,
                        random_state=self.random_state,
                        metric=self.metric,
                        parimony_coefficient=self.parsimony_coefficient,
                        program=offspring_program
                    )
                    
                    new_population.append(offspring)
            
            # Replace the old population
            self.population = new_population
            
            # Update tracking information
            self._update_best()
            
            # Store generation statistics
            generation_stats = {
                'generation': gen,
                'best_fitness': self.best_fitness,
                'avg_fitness': np.mean([self._evaluate_fitness(p) for p in self.population]),
                'best_length': len(self.best_program.program),
                'avg_length': np.mean([len(p.program) for p in self.population]),
                'fitness_evaluations': self.fitness_evaluations,
                'elapsed_time': time.time() - self.start_time
            }
            
            self.history.append(generation_stats)
            
            # Log progress
            if verbose and (gen % log_interval == 0 or gen == self.generations - 1):
                elapsed = time.time() - self.start_time
                remaining = (elapsed / (gen + 1)) * (self.generations - gen - 1)
                
                print(f"Generation {gen+1}/{self.generations} | "
                      f"Best Fitness: {self.best_fitness:.4f} | "
                      f"Avg Fitness: {generation_stats['avg_fitness']:.4f} | "
                      f"Best Length: {generation_stats['best_length']} | "
                      f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                      f"Remaining: {timedelta(seconds=int(remaining))}")
        
        # Final update
        self._update_best()
        
        if verbose:
            print(f"\nEvolution completed in {timedelta(seconds=int(time.time() - self.start_time))}")
            print(f"Total fitness evaluations: {self.fitness_evaluations}")
            print(f"Best fitness achieved: {self.best_fitness:.4f}")
            print(f"Best program length: {len(self.best_program.program)}")
            
        return self.best_program
    
    def get_best_individual(self):
        """Return the best individual found so far."""
        return self.best_program
    
    def get_fitness_history(self):
        """Return the history of best fitness values."""
        return [stats['best_fitness'] for stats in self.history]
    
    def get_all_history(self):
        """Return all tracked statistics."""
        return self.history





