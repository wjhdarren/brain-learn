import requests
import numpy as np
import time
import os
from src.utils import evaluate_fitness
from datetime import timedelta
from src.program import Program
from itertools import batched 
import concurrent.futures
import threading
import warnings

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
                 parsimony_coefficient=0.15,
                 n_parallel=3,
                 init_population=[]):
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
        n_parallel : int, optional
            Number of parallel workers for fitness evaluation (default is 1). Max is 3.
        init_population : list, optional
            List of Program objects to initialize the population with.
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
        self.n_parallel = n_parallel
        
        # Setup random state
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state
            
        # Initialize tracking variables
        self.population = [Program(max_depth=self.max_depth,
                                max_operators=self.max_operators,
                                random_state=self.random_state,
                                metric=self.metric,
                                parimony_coefficient=self.parsimony_coefficient, 
                                program=program) for program in init_population]
        self.history = []
        self.best_program = None
        self.best_fitness = float('-inf')
        self.generation = 0
        
        # Statistics
        self.fitness_evaluations = 0
        self.start_time = None
        self._session_lock = threading.Lock()
        
    def _initialize_population(self):
        """Initialize a population of random programs."""
        while len(self.population) < self.population_size:
            program = Program(
                max_depth=self.max_depth,
                max_operators=self.max_operators,
                random_state=self.random_state,
                metric=self.metric,
                parimony_coefficient=self.parsimony_coefficient
            )
            self.population.append(program)
            if len(self.population) == self.population_size:
                break
        else:
            self.population = self.random_state.choice(self.population, size=self.population_size, replace=False)
    
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
    
    def _evaluate_single_program(self, program):
        """
        Evaluate the fitness of a single program, handling retries and session recreation.

        Returns
        -------
        tuple
            (raw_fitness, fitness, needs_evaluation_increment)
        """
        if program.fitness is not None:
            # Already evaluated
            return program.raw_fitness, program.fitness, False

        needs_evaluation_increment = True
        fast_expr = program.__str__()
        raw_fitness = float('-inf')

        for attempt in range(MAX_RECONNECTION_ATTEMPTS):
            try:
                result = self.metric(fast_expr)

                if result is None:
                    # Authentication likely failed, try to recreate session
                    recreated = False
                    with self._session_lock: # Ensure only one thread recreates session
                        # Double check if another thread already recreated it while waiting for the lock
                        try:
                            test_response = self.session.get('https://api.worldquantbrain.com/authentication') # A lightweight check
                            if test_response.status_code != 200:
                                recreated = self._recreate_session()
                            else: # Session seems okay now
                                recreated = True # Act as if recreated to retry metric call
                        except requests.exceptions.RequestException:
                             recreated = self._recreate_session() # Error during check, try recreating

                    if recreated:
                        continue # Retry metric call with potentially new session
                    else:
                        raw_fitness = float('-inf')
                        break # Failed to recreate session

                fitness = result.get('fitness')
                raw_fitness = fitness if fitness is not None else float('-inf')
                # Append program to initial-population.pkl if thresholds are met
                try:
                    if ((result.get('LOW_SHARPE') == 'PASS' and result.get('LOW_FITNESS') == 'PASS' and result.get('LOW_TURNOVER') == 'PASS' and result.get('HIGH_TURNOVER') == 'PASS')
                        or (result.get('sharpe', 0) > 1.7)
                        or (result.get('fitness', 0) >= 1.1)):
                        import dill
                        try:
                            with open('initial-population.pkl', 'rb') as f:
                                population = dill.load(f)
                        except (FileNotFoundError, EOFError):
                            population = []
                        population.append(program.program)
                        with open('initial-population.pkl', 'wb') as f:
                            dill.dump(population, f)
                except Exception as e:
                    print(f"Error appending program to initial-population.pkl: {e}")
                break # Success

            except Exception as e:
                error_message = str(e).lower()
                if "401" in error_message or "authentication" in error_message:
                    # Authentication error, attempt to reconnect
                    reconnected = False
                    with self._session_lock:
                       # Double check session status before attempting recreation
                        try:
                            test_response = self.session.get('https://api.worldquantbrain.com/authentication')
                            if test_response.status_code != 200:
                                reconnected = self._recreate_session()
                            else:
                                reconnected = True
                        except requests.exceptions.RequestException:
                            reconnected = self._recreate_session()

                    if reconnected:
                        continue # Retry metric call
                elif "429" in error_message or "simulation_limit_exceeded" in error_message:
                    # Rate limiting error, wait and retry
                    print("Rate limit exceeded. Waiting before retry...")
                    # Wait with exponential backoff
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                    time.sleep(wait_time)
                    continue  # Retry after waiting

                # For other errors or failed reconnection, assign -inf and maybe log
                raw_fitness = float('-inf')
                if attempt == MAX_RECONNECTION_ATTEMPTS - 1:
                    print(f"Failed to evaluate fitness for program after {MAX_RECONNECTION_ATTEMPTS} attempts: {e}")
                # Consider breaking here unless it was an auth error we might recover from on retry
                # For simplicity, we break on any exception after checking auth
                break


        # Calculate final fitness with parsimony pressure
        fitness = raw_fitness - program.parimony_coefficient * len(program.program)

        return raw_fitness, fitness, needs_evaluation_increment


    def _evaluate_fitness(self, program):
        """Evaluate the fitness of a program (sequential wrapper)."""
        raw_fitness, fitness, increment = self._evaluate_single_program(program)

        # Update program state if it was evaluated now
        if increment:
            program.raw_fitness = raw_fitness
            program.fitness = fitness
            self.fitness_evaluations += 1

        return program.fitness # Return the fitness value

    def parallel_evaluate_fitness(self, programs_to_evaluate, n_parallel=3):
        """
        Evaluate the fitness of multiple programs in parallel.

        Parameters
        ----------
        programs_to_evaluate : list
            A list of Program objects to evaluate.
        n_parallel : int, optional
            Number of parallel workers (default is 3, max is 3).

        Returns
        -------
        list
            The list of programs with their fitness attributes updated.
        """
        if n_parallel > 3:
            warnings.warn(
                f"Requested n_parallel={n_parallel} exceeds the limit of 3. Capping at 3.",
                UserWarning,
                stacklevel=2
            )
            n_parallel = 3
        elif n_parallel <= 0:
            raise ValueError("n_parallel must be positive.")

        if not programs_to_evaluate:
            return []

        total_evaluations_increment = 0
        
        BATCH_SIZE = 50 # Using batch processing for better throughput
        TIMEOUT_PER_PROGRAM = 180  # seconds
        
        for batch_idx, batch in enumerate(batched(programs_to_evaluate, BATCH_SIZE)):
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
                # Submit batch of tasks
                future_to_program = {executor.submit(self._evaluate_single_program, p): p for p in batch}
                
                # Process completed futures with timeout
                timeout = TIMEOUT_PER_PROGRAM * len(batch) / n_parallel
                
                try:
                    for future in concurrent.futures.as_completed(future_to_program, timeout=timeout):
                        program = future_to_program[future]
                        try:
                            raw_fitness, fitness, increment = future.result()
                            # Update program state directly
                            program.raw_fitness = raw_fitness
                            program.fitness = fitness
                            if increment:
                                total_evaluations_increment += 1
                        except Exception as exc:
                            print(f'{program!r} generated an exception: {exc}')
                            # Handle failed evaluations
                            program.raw_fitness = float('-inf')
                            program.fitness = float('-inf')
                except concurrent.futures.TimeoutError:
                    print("Timeout reached while evaluating batch. Some programs might not have been evaluated.")
                    # Mark incomplete programs with -inf fitness
                    for future, program in future_to_program.items():
                        if not future.done():
                            program.raw_fitness = float('-inf')
                            program.fitness = float('-inf')
            
            # Add a delay between batches to help prevent rate limiting
            # Only add delay if there are more batches to process
            if batch_idx < len(list(batched(programs_to_evaluate, BATCH_SIZE))) - 1:
                # Wait between batches to avoid hitting rate limits
                # The delay increases with each batch to help with rate limiting
                batch_delay = 5 + batch_idx  # 5, 6, 7, ... seconds
                print(f"Waiting {batch_delay} seconds before processing next batch to avoid rate limits...")
                time.sleep(batch_delay)

        # Update the global counter after all batches are done
        self.fitness_evaluations += total_evaluations_increment

        # Final check for any programs without fitness values
        for p in programs_to_evaluate:
            if p.fitness is None:
                p.raw_fitness = float('-inf')
                p.fitness = float('-inf')

        return programs_to_evaluate
    
    def _tournament_selection(self):
        """Select an individual using tournament selection. Assumes fitness is pre-calculated."""
        indices = self.random_state.randint(0, len(self.population), self.tournament_size)
        tournament = [self.population[i] for i in indices]
        
        # Return the best individual in the tournament based on pre-calculated fitness
        # Handle potential None fitness values gracefully if evaluation failed.
        return max(tournament, key=lambda program: program.fitness if program.fitness is not None else float('-inf'))
    
    def _update_best(self):
        """Update the best program seen so far. Assumes fitness is pre-calculated."""
        # Find the best program based on the pre-calculated fitness
        current_best = max(self.population, key=lambda program: program.fitness if program.fitness is not None else float('-inf'))
        current_best_fitness = current_best.fitness if current_best.fitness is not None else float('-inf')
        
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
            # Evaluate fitness of the initial population
            self.parallel_evaluate_fitness(self.population, self.n_parallel)
            self._update_best() # Initial best based on initial population
            
        self.start_time = time.time()
        
        for gen in range(self.generations):
            self.generation = gen
            
            # Create a new generation
            new_population = []
            
            # Elite selection - keep the best individual (fitness already calculated)
            elite = max(self.population, key=lambda program: program.fitness if program.fitness is not None else float('-inf'))
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
            
            # Evaluate fitness for the new generation in parallel (or sequentially if n_parallel=1)
            self.parallel_evaluate_fitness(self.population, self.n_parallel)
            
            # Update tracking information using pre-calculated fitness
            self._update_best()
            
            # Store generation statistics
            valid_fitnesses = [p.fitness for p in self.population if p.fitness is not None]
            avg_fitness = np.mean(valid_fitnesses) if valid_fitnesses else float('-inf')
            generation_stats = {
                'generation': gen,
                'best_fitness': self.best_fitness,
                'avg_fitness': avg_fitness, # Use pre-calculated fitness
                'best_length': len(self.best_program.program) if self.best_program else 0, # Handle case where best_program might not be set yet
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
        
        # Final update using last generation's fitness
        # No need to call _update_best() again as it was called after the last parallel_evaluate_fitness
        
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





