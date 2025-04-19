from src.function import * # noqa: F403
import numpy as np
from numpy.random import RandomState
from copy import deepcopy

class Program:
    def __init__(
        self,
        max_depth : int,
        max_operators : int,
        random_state : RandomState,
        metric : callable,
        parimony_coefficient : float = 0.1,
        p_point_replace : float = 0.1,
        program : list[Operator | Terminal] = None,
        skip_validation : bool = False, # for debug usage
    ):
        self.max_depth = max_depth
        self.max_operators = max_operators
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parimony_coefficient = parimony_coefficient
        self.random_state = random_state
        self.program = program
        self.arities = ARITIES
        self.operator_set = OPERATORS 
        self.terminal_set = TERMINALS # import from function.py

        if self.program is None:
            # Create a naive random program if none provided
            self.program = self.build_program(random_state)
        elif self.program and not skip_validation and not self.validate_program():  # Only validate non-empty programs
            raise ValueError('The supplied program is incomplete.')
        
        self.raw_fitness = None
        self.fitness = None
        self._unit = None
        self.parents = None
    
        
    @property
    def unit(self):
        """Compute and return the units for each node in the program.
        
        Returns
        -------
        list
            A list of sympy symbolic values representing the units of each node.
        """
        if self._unit is None or len(self._unit) != len(self.program):
            self._unit = [None] * len(self.program)
            
            # Use a stack to track units during program evaluation
            stack = []
            
            for i, node in enumerate(self.program):
                if isinstance(node, Terminal):
                    # For terminal nodes, use their predefined unit
                    self._unit[i] = node.unit
                    stack.append(node.unit)
                elif isinstance(node, Operator):
                    # For operators, pop required number of operands
                    if len(stack) < node.arity:
                        raise ValueError(f'Not enough operands for operator {node.name}')
                    
                    # Get operand units from the stack (in reverse order for proper handling)
                    operand_units = []
                    for _ in range(node.arity):
                        operand_units.insert(0, stack.pop())
                    
                    # Apply the unit rule and store the result
                    try:
                        result_unit = node.unit_rule(*operand_units)
                        self._unit[i] = result_unit
                        stack.append(result_unit)
                    except Exception as e:
                        operand_str = ', '.join([str(u) for u in operand_units])
                        raise ValueError(f'Unit rule failed for operator {node.name} with units [{operand_str}]: {str(e)}') from e
                else:
                    raise ValueError(f'Unknown node type: {node}')
            
            # Validate that we end up with exactly one unit on the stack
            if len(stack) != 1:
                raise ValueError(f'Invalid program structure: expected 1 final unit but got {len(stack)}')
            
        return self._unit  
            
    def build_program(self, random_state):
        """Build a random program tree in postfix notation.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program in postfix notation.
        """
        max_attempts = 50  # Maximum number of attempts to build a valid program
        max_depth = int(2/3*self.max_depth)
        max_operators = int(2/3*self.max_operators)
        
        def weighted_choice(options, weights=None):
            """Helper function for weighted random choice."""
            if not options:
                return None
                
            if weights is None:
                weights = np.array([getattr(option, 'weight', 1.0) for option in options])
                
            if weights.sum() == 0:
                weights = np.ones_like(weights)
                
            probs = weights / weights.sum()
            choice_idx = random_state.choice(len(options), p=probs)
            return options[choice_idx]
        
        # Function to generate a random subprogram
        def generate_subprogram(depth=0, remaining_operators=None, min_depth=0):
            if remaining_operators is None:
                remaining_operators = max_operators
                
            # Determine if we should use a terminal or operator
            # Probability of using a terminal increases with depth and decreases with remaining operators
            p_terminal = 0.1 + 0.7 * min(1.0, depth / max_depth)
            
            # Force operators for the first few levels to ensure some complexity
            if depth < min_depth:
                p_terminal = 0.0
                
            if depth >= max_depth or remaining_operators <= 0 or (depth >= min_depth and random_state.uniform() < p_terminal):
                # Choose terminal
                terminal = weighted_choice(self.terminal_set)
                return [terminal], 0
            else:
                # Choose operator
                if random_state.uniform() < 0.7 and self.arities.get(2, []):  # Prefer binary operators
                    operator = weighted_choice(self.arities[2])
                else:
                    operator = weighted_choice(self.operator_set)
                
                # Generate operands for this operator
                program = []
                for _ in range(operator.arity):
                    # Encourage more depth on the first operand for binary operators
                    next_min_depth = 0
                    if operator.arity == 2 and _ == 0 and depth < 2:
                        next_min_depth = 1
                        
                    subprogram, sub_operators = generate_subprogram(
                        depth + 1,
                        remaining_operators - 1 - len(program) // 2,  # Rough estimate of remaining operators
                        min_depth=next_min_depth
                    )
                    program.extend(subprogram)
                    
                # Add the operator after its operands (postfix notation)
                program.append(operator)
                return program, len([node for node in program if isinstance(node, Operator)])
        
        # Try to generate valid programs
        for _ in range(max_attempts):
            # Generate a random program - start with at least 2 levels of depth
            try:
                program, _ = generate_subprogram(min_depth=min(2, self.max_depth // 2))
                
                # Save temporary program to validate it
                temp_program = self.program
                self.program = program
                
                # Check if the program is valid
                if self.validate_program():
                    return program
                
                # Restore original program if validation fails
                self.program = temp_program
            except Exception:
                # Just try again on any exception
                continue
                
        # If all attempts failed, try with a simple program
        for _ in range(10):
            try:
                if 2 in self.arities and self.arities[2]:
                    # Try a simple binary operation
                    operator = weighted_choice(self.arities[2])
                    terminal1 = weighted_choice(self.terminal_set)
                    terminal2 = weighted_choice(self.terminal_set)
                    program = [terminal1, terminal2, operator]
                    
                    # Validate and return if valid
                    temp_program = self.program
                    self.program = program
                    if self.validate_program():
                        return program
                    self.program = temp_program
            except Exception:
                continue
                
        # Last resort fallback to a single terminal
        terminal = weighted_choice(self.terminal_set)
        return [terminal]

    def validate_program(self):
        """Validate that the embedded program in the object is valid.
        For a valid postfix expression, we should end up with exactly one value on the stack.
        Also validates unit compatibility for all operations.
        Also checks that the program doesn't exceed max_depth and max_operators constraints.
        """
        # Create a stack to simulate evaluation
        value_stack = []
        unit_stack = []
        
        # # Check max_operators constraint 
        # if self.operator_count() > self.max_operators:
        #     return False
        
        # # Check max_depth constraint 
        # if self.depth() > self.max_depth:
        #     return False
        
        for _, node in enumerate(self.program):
            if isinstance(node, Terminal):
                # Push terminal nodes onto the stack
                value_stack.append(node)
                unit_stack.append(node.unit)
            elif isinstance(node, Operator):
                # Check if we have enough operands for this operator
                if len(value_stack) < node.arity:
                    return False
                
                # Pop the required number of operands for values
                operands = []
                for _ in range(node.arity):
                    operands.insert(0, value_stack.pop())
                
                # Pop the required number of units
                unit_operands = []
                for _ in range(node.arity):
                    unit_operands.insert(0, unit_stack.pop())
                
                # Check if the unit rule can be applied successfully
                try:
                    result_unit = node.unit_rule(*unit_operands)
                except Exception:
                    # Failed unit rule indicates an invalid program
                    return False
                
                # Push the result back onto the stacks
                value_stack.append(None)  # Placeholder for the result
                unit_stack.append(result_unit)
            else:
                return False  # Unknown node type
        
        # A valid program should leave exactly one result on the stack
        return len(value_stack) == 1 and len(unit_stack) == 1
    
    def __str__(self):
        if not self.program:
            return 'EmptyProgram'
        
        # Stack for operands and results
        stack = []
        
        for node in self.program:
            if isinstance(node, Terminal):
                # For terminal nodes, push their name to the stack (not value)
                stack.append(str(node.name))
            elif isinstance(node, Operator):
                # For operators, pop required number of operands
                if len(stack) < node.arity:
                    raise ValueError(f'Not enough operands for operator {node.name}')
                
                # Get operands from the stack (in reverse order)
                operands = []
                for _ in range(node.arity):
                    operands.insert(0, stack.pop())
                
                # Apply the operator function and push result back to stack
                try:
                    result = node.function(*operands)
                    stack.append(result)
                except Exception as e:
                    child_vals = ', '.join([f"'{s}'" for s in operands])
                    raise TypeError(f'Error formatting operator {node.name} with arguments [{child_vals}]: {str(e)}') from e
            else:
                raise ValueError(f'Unknown node type: {node}')
        
        # We should have exactly one item on the stack - the final result
        if len(stack) != 1:
            raise ValueError(f'Invalid program structure: expected 1 result but got {len(stack)}')
        
        return stack[0]
   
    def depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, Operator):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1
    
    def operator_count(self):
        return sum(1 for node in self.program if isinstance(node, Operator))
    
    def length(self):
        return len(self.program)

    def raw_fitness(self):
        fast_expr = self.__str__()
        try:
            sharpe = self.metric(fast_expr)['sharpe']
            if sharpe is None:
                return float('-inf')
            return sharpe
        except Exception:
            return float('-inf')

    def fitness(self):
        penalty = self.parimony_coefficient * len(self.program) 
        return self.raw_fitness - penalty
    
    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.
        """
        if program is None:
            program = self.program
        
        if not program:  # Handle empty programs
            return 0, 0
        
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array(
            [0.9 if isinstance(node, Operator) else 0.1 for node in program]
        )
        
        # Ensure probabilities sum to 1
        if probs.sum() > 0:
            probs = np.cumsum(probs / probs.sum())
        else:
            probs = np.ones(len(program)) / len(program)
            probs = np.cumsum(probs)
        
        # Select start point
        start = np.searchsorted(probs, random_state.uniform())
        
        # Ensure start is in bounds
        start = min(start, len(program) - 1)

        # Find end of subtree
        stack = 1
        end = start
        
        # Ensure we don't go out of bounds
        while stack > end - start and end < len(program):
            node = program[end]
            if isinstance(node, Operator):
                stack += node.arity
            end += 1
        
        # If we couldn't find a complete subtree, just return the node itself
        if end >= len(program) and stack > end - start:
            return start, start + 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return deepcopy(self.program)
    
    def crossover(self, donor, random_state = None):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.
        removed : list
            Indices of nodes removed from the original program.
        donor_removed : list
            Indices of nodes removed from the donor program.
        """
        # Maximum attempts to find a valid crossover
        max_attempts = 10
        if random_state is None:
            random_state = self.random_state
            
        for _ in range(max_attempts):
            # Get a subtree to replace
            start, end = self.get_subtree(random_state)
            removed = range(start, end)
            
            # Get a subtree to donate
            donor_start, donor_end = self.get_subtree(random_state, donor)
            donor_removed = list(
                set(range(len(donor))) - set(range(donor_start, donor_end))
            )
            
            # Create the new program by inserting genetic material from donor
            new_program = (
                self.program[:start] + donor[donor_start:donor_end] + self.program[end:]
            )
            
            # Check if the new program is valid (including unit compatibility)
            temp_program = Program(
                max_depth=self.max_depth,
                max_operators=self.max_operators,
                metric=self.metric,
                random_state=random_state,
                parimony_coefficient=self.parimony_coefficient,
                p_point_replace=self.p_point_replace,
                program=new_program,
                skip_validation=True  # Skip initial validation
            )
            
            # Validate the program including unit compatibility
            if temp_program.validate_program():
                return new_program, removed, donor_removed
        
        # If no valid crossover was found, return a copy of the original program
        return deepcopy(self.program), [], []
    
    def subtree_mutation(self, random_state = None):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if random_state is None:
            random_state = self.random_state
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)
    
    def hoist_mutation(self, random_state = None):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.
        removed : list, optional
            The indices of nodes that were removed from the original program.
        """
        # Maximum attempts to find a valid mutation
        max_attempts = 10
        
        if random_state is None:
            random_state = self.random_state
        
        for _ in range(max_attempts):
            # Get a subtree to replace
            start, end = self.get_subtree(random_state)
            subtree = self.program[start:end]
            
            # Get a subtree of the subtree to hoist
            sub_start, sub_end = self.get_subtree(random_state, subtree)
            hoist = subtree[sub_start:sub_end]
            
            # Create the new program by replacing the original subtree with the hoisted subtree
            new_program = self.program[:start] + hoist + self.program[end:]
            
            # Determine which nodes were removed for plotting
            removed = list(
                set(range(start, end)) - set(range(start + sub_start, start + sub_end))
            )
            
            # Check if the new program is valid (including unit compatibility)
            temp_program = Program(
                max_depth=self.max_depth,
                max_operators=self.max_operators,
                metric=self.metric,
                random_state=random_state,
                parimony_coefficient=self.parimony_coefficient,
                p_point_replace=self.p_point_replace,
                program=new_program,
                skip_validation=True  # Skip initial validation
            )
            
            # Validate the program including unit compatibility
            if temp_program.validate_program():
                return new_program, removed
        
        # If no valid mutation was found, return a copy of the original program
        return deepcopy(self.program), []


    def point_mutation(self, random_state = None):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.
        mutated : list
            Indices of nodes that were mutated.
        """
        # Maximum attempts to find a valid mutation
        max_attempts = 10
        if random_state is None:
            random_state = self.random_state
        
        for _ in range(max_attempts):
            program = deepcopy(self.program)

            # Get the nodes to modify
            mutate = np.where(
                random_state.uniform(size=len(program)) < self.p_point_replace
            )[0]

            for node in mutate:
                if isinstance(program[node], Operator):
                    arity = program[node].arity
                    # Find a valid replacement with same arity
                    if arity in self.arities and self.arities[arity]:
                        valid_replacements = self.arities[arity]
                        replacement_idx = random_state.randint(len(valid_replacements))
                        program[node] = valid_replacements[replacement_idx]
                else:
                    # Replace terminal with another terminal from terminal_set
                    if self.terminal_set:
                        terminal_idx = random_state.randint(len(self.terminal_set))
                        program[node] = self.terminal_set[terminal_idx]

            # Check if the new program is valid (including unit compatibility)
            temp_program = Program(
                max_depth=self.max_depth,
                max_operators=self.max_operators,
                metric=self.metric,
                random_state=random_state,
                parimony_coefficient=self.parimony_coefficient,
                p_point_replace=self.p_point_replace,
                program=program,
                skip_validation=True  # Skip initial validation
            )
            
            # Validate the program including unit compatibility
            if temp_program.validate_program():
                return program, list(mutate)
        
        # If no valid mutation was found, return a copy of the original program
        return deepcopy(self.program), []
    
    @staticmethod
    def create_from_list(program : list[Operator | Terminal]):
        return Program(
            max_depth=len(program),
            max_operators=len(program),
            metric=lambda x: 0,
            random_state=np.random.RandomState(),
            program=program,
        )
    
    depth_ = property(depth)
    operator_count_ = property(operator_count)
    length_ = property(length)  
    
    