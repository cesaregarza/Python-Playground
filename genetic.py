import numpy as np
import random

class Genetic_Vector:
    def __init__(self, vector, generation, parameter_bounds, fitness_function, mutation_rate = 0.015, simple_mutation = True, simple_crossover = True, crossover_function = None, type_check = True, creep_mutate = True):
        self.vector = vector
        self.generation = generation
        self.parameter_bounds = parameter_bounds
        self.fitness = fitness_function
        self.simple_mutation = simple_mutation
        self.mutation_rate = mutation_rate
        self.simple_crossover = simple_crossover
        self.crossover_function = crossover_function
        self.creep_mutate = creep_mutate
        self.fitness_val = self.fitness(*list(self.vector))
        self.vector_length = len(self.vector)

        if type_check:
            if not isinstance(self.vector, np.ndarray):
                self.vector = np.squeeze(np.asarray(self.vector))
                if len(self.vector.shape) > 1:
                    raise TypeError("Parameter vector must be a 1-dimensional array-like")
            
            if isinstance(parameter_bounds, tuple) and len(parameter_bounds) == 2:
                self.parameter_bounds = [parameter_bounds] * self.vector_length
            
            if not all(isinstance(item, tuple) and len(item) == 2 for item in self.parameter_bounds):
                raise TypeError("Parameter parameter_bounds must be a list of 2-tuples")
            
            if not callable(self.fitness):
                raise TypeError("Parameter fitness_function must be a function")

            if not simple_crossover and crossover_function is None:
                raise TypeError("Declare a crossover function, otherwise flag simple_crossover as True")
    
    def __lt__(self, other):
        return self.fitness_val < other.fitness_val
    
    def __le__(self, other):
        return self.fitness_val <= other.fitness_val
    
    def __eq__(self, other):
        return self.fitness_val == other.fitness_val
    
    def __gt__(self, other):
        return self.fitness_val > other.fitness_val
    
    def __ge__(self, other):
        return self.fitness_val >= other.fitness_val
    
    def __ne__(self, other):
        return self.fitness_val != other.fitness_val
        
    def mate(self, other):
        """Mating Function
        
        Parameters
        -----------
        other: Genetic_Vector
            The other parent
        
        Returns
        -------
        child: Genetic_Vector
            Child vector
        """
        if not isinstance(other, Genetic_Vector):
            raise ValueError("Other parent is invalid")
        
        if self.vector_length != other.vector_length:
            raise ValueError("Vector lengths do not match")
        
        child_vector = np.empty(self.vector_length)
        
        #If we are doing simple crossover, we're just going to take one of the two parent parameters
        if self.simple_crossover:
            for i in range(self.vector_length):
                parents = [self, other]
                rand_parent = random.randint(0,1)
                child_vector[i] = parents[rand_parent].vector[i]
        #If we are not doing simple crossover, we're going to evaluate the complex crossover
        else:
            for i in range(self.vector_length):
                parent_param_tuple = (self.vector[i], other.vector[i])
                param_bounds = self.parameter_bounds[i]
                child_vector[i] = self._complex_crossover(parent_param_tuple, param_bounds)
        
        child_mutation_rate = self.mutation_rate + (random.random() * (0.4) - 0.2) * self.mutation_rate

        child_vector = self.mutate(child_vector, child_mutation_rate)
        child_generation = max(self.generation, other.generation) + 1
        
        child = Genetic_Vector(child_vector, child_generation, self.parameter_bounds, self.fitness, mutation_rate = child_mutation_rate, simple_mutation = self.simple_mutation, simple_crossover = self.simple_crossover, crossover_function = self.crossover_function, type_check = False)

        return child

    def mutate(self, vector, mutation_rate):
        """Mutation Function

        Parameters
        ----------
        vector: array
            The vector that will undergo mutation
        
        mutation_rate: float
            The mutation rate
        
        Returns
        -------
        vector: array
            vector that has underwent mutation (if at all)
        """
        if not self.creep_mutate:
            for i in range(len(vector)):
                r = random.random()
                if r < mutation_rate:
                    vector[i] = random.randint(*self.parameter_bounds[i])
            
            return vector
        else:
            for i in range(len(vector)):
                lower, upper = self.parameter_bounds[i]
                r = random.random()
                s = random.random() * (upper - lower) * 0.25 - (upper - lower) * 0.5
                if r < mutation_rate:
                    if vector[i] + s < upper and vector[i] + s > lower:
                        vector[i] += s
            
            return vector
    
    def _complex_crossover(self, params, specific_bounds):
        """Custom crossover function

        Parameters
        ----------
        param: 2-tuple
            The appropriate parameter for both parents
        
        specific_bounds: 2-tuple
            Upper and lower bounds for the parameter
        
        Returns
        -------
        child_value: value
            Result of crossover function
        """
        candidate_within_bounds = False
        #Loop so long as the candidate is not within the bounds 
        while not candidate_within_bounds:
            candidate_value = self.crossover_function(params)
            lower_bound, upper_bound = specific_bounds
            candidate_within_bounds = (candidate_value >= lower_bound) and (candidate_value <= upper_bound)
        
        return candidate_value

class Genetic_Pool:
    def __init__(self, template, size = None, elitism = False, selection_method = "tournament"):
        if not isinstance(template, Genetic_Vector):
            raise TypeError("Parameter template must be a Genetic_Vector object")

        self.template = template
        self.elitism = elitism
        self.members = (self.template.vector_length * 10) if not size else size
        self.selection_method = selection_method
        self.pool = []
    
    def _parent_selection(self):
        fxn_dict = {
            "tournament": self._tournament
        }
        try:
            optimization_chosen = fxn_dict[self.selection_method.lower()]
        except KeyError:
            raise ValueError(f"{self.selection_method} is not a valid optimization method")

        return optimization_chosen()
        
    def _tournament(self, tournament_size = 6):
        """Runs a tournament knockout style for parent selection
        
        Keyword Arguments:
            tournament_size {int} -- Number of entrants to a tournament. The farther from a power of 2, the more likely it is to have diversity (default: {6})
        
        Returns:
            Genetic_Vector[] -- List of Genetic_Vectors that are the selected parents
        """
        
        try:
            candidates = random.sample(self.pool, tournament_size)
        except ValueError:
            return random.sample(self.pool, 2)
        
        rounds = (tournament_size - 1).bit_length() - 1

        for i in range(rounds):
            next_round = []
            while len(candidates) > 1:
                random.shuffle(candidates)
                fighters = candidates[:2]
                candidates = candidates[2:]
                next_round.append(max(fighters))

            candidates += next_round
        
        return candidates
    
    def new_generation(self):
        """Creates a new generation
        
        Raises:
            RuntimeError: self.initialize() must be ran at least once before using this function
        
        Returns:
            Genetic_Vector[] -- List of Genetic Vectors
        """
        gen_size = self.members
        next_gen = []
        if len(self.pool) == 0:
            raise RuntimeError("New Generation cannot be generated without first initializing using the initialize() method")
        
        if self.elitism:
            next_gen.append(self.pool[-1])
            self.pool = self.pool[:-1]
        
        while len(next_gen) < gen_size:
            parents = self._parent_selection()
            try:
                next_gen.append(parents[0].mate(parents[1]))
            except TypeError as e:
                print(vars(parents[0]))
                print(vars(parents[1]))
                raise TypeError(e)
        
        self.pool = next_gen
        self.pool.sort()
    
    def initialization(self, generation_function = None):
        """Initializes the Genetic_Pool
        
        Keyword Arguments:
            generation_function {function} -- If provided, uses provided function to generate the initial population (default: {None})
        
        Returns:
            None -- No return
        """
        if generation_function is None:
            def generation_function():
                vector_list = []
                for lower, upper in self.template.parameter_bounds:
                    distance = upper - lower
                    
                    if isinstance(lower, float):
                        value = random.random() * distance + lower
                    else:
                        value = random.randint(lower, upper)
                    
                    vector_list.append(value)
                new_genetic_vector = Genetic_Vector(vector_list, 1,self.template.parameter_bounds, self.template.fitness, mutation_rate = self.template.mutation_rate, simple_mutation = self.template.simple_mutation, simple_crossover=self.template.simple_crossover, crossover_function = self.template.crossover_function, type_check = False)
                return new_genetic_vector
        
        for i in range(self.members):
            self.pool.append(generation_function())
        
        self.pool.sort()
        self.generation_max = 0
        self.score_max = self.pool[-1].fitness_val
    
    def initialize(self, generation_function):
        self.initialization(generation_function)
    
    def multiple_generations(self, generations, early_stopping = True, es_percent_generations = 0.3):
        """Runs multiple generations
        
        Arguments:
            generations {int} -- Number of generations you want to run
        
        Keyword Arguments:
            early_stopping {bool} -- Whether or not to use early stopping (default: {True})
            es_percent_generations {float} -- Early Stopping Percent of Generations. If early_stopping is true, how much of the generation without changes should indicate stopping (default: {0.3})
        
        Returns:
            [type] -- [description]
        """
        for i in range(generations):
            self.new_generation()
            if self.pool[-1].fitness_val > self.score_max:
                self.score_max = self.pool[-1].fitness_val
                self.generation_max = i + 1
            elif early_stopping:
                if (i - self.generation_max) > (es_percent_generations * generations):
                    print(f"Stopped at generation {i}")
                    break
        
        return self.pool[-1]
