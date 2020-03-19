import numpy as np
import random
import fast_random as frandom
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Genetic_Template:
    def __init__(self, parameter_bounds, fitness_function, mutation_rate = 0.25, mutation = True, simple_crossover = True, crossover_function = None, creep_mutate = True):
        """Creates a Genetic Template object
        
        Arguments:
            parameter_bounds {(float/int, float/int)[]} -- List of upper/lower bounds for each hyperparameter.
            fitness_function {function} -- Function that will be used to evaluate fitness. Genetic Algorithm seeks greatest fitness, so to find minimums transform using f(x) = -x
        
        Keyword Arguments:
            mutation_rate {float} -- Rate at which mutation, if enabled, will occur (default: {0.25})
            mutation {bool} -- Mutate vector on crossover (default: {True})
            simple_crossover {bool} -- use built-in crossover function (default: {True})
            crossover_function {function} -- User-defined function used to control crossover (default: {None})
            creep_mutate {bool} -- Enable Creep Mutation (default: {True})
        
        Raises:
            TypeError: Raised if parameter_bounds is not a list of 2-tuples
            TypeError: Raised if fitness_function is not a function
            TypeError: Raised if simple_crossover is disabled and crossover_function is not defined
        """
        self.parameter_bounds = parameter_bounds
        self.fitness = fitness_function
        self.mutation_rate = mutation_rate
        # self.simple_mutation = simple_mutation
        self.vector_length = len(self.parameter_bounds)
        self.mutation = mutation
        self.simple_crossover = simple_crossover
        self.crossover_function = crossover_function
        self.creep_mutate = creep_mutate
        
        if isinstance(parameter_bounds, tuple) and len(parameter_bounds) == 2:
            self.parameter_bounds = [parameter_bounds] * self.vector_length
        
        if not all(isinstance(item, tuple) and (len(item) == 2 or len(item) == 3) for item in self.parameter_bounds):
            raise TypeError("Parameter parameter_bounds must be a list of 2-tuples")
        
        if not callable(self.fitness):
            raise TypeError("Parameter fitness_function must be a function")

        if not simple_crossover and crossover_function is None:
            raise TypeError("Declare a crossover function, otherwise flag simple_crossover as True")

class Genetic_Vector(Genetic_Template):
    def __init__(self, vector, generation, template, type_check = True, penalty_check = False):
        """Creates a Genetic Vector object
        
        Arguments:
            Genetic_Template {Genetic_Template} -- Template used to create vectors
            vector {float[]} -- List of hyperparameters to be optimized
            generation {int} -- Generation of the Genetic Vector
            template {Genetic_Template} -- Template to be used (used for referential reasons)
        
        Keyword Arguments:
            type_check {bool} -- Whether type checking should be enabled. Should only be disabled for optimization (default: {True})
            penalty_check {bool} -- Apply a fitness penalty if resulting vector lies outside given parameter bounds (default: {False})
        
        Raises:
            TypeError: Raised if vector provided is not a 1-dimensional array-like
        """
        self.vector = vector
        self.generation = generation
        self.template = template
        # self.simple_mutation = template.simple_mutation
        self.fitness_val = self.template.fitness(*self.vector)
        self.vector_length = len(self.vector)
        
        if penalty_check:
            for i in range(self.vector_length):
                if self.vector[i] < self.template.parameter_bounds[i][0] or self.vector[i] > self.template.parameter_bounds[i][1]:
                    self.fitness_val *= 10

        if type_check and not isinstance(self.vector, np.ndarray):
                temp_vector = np.squeeze(np.asarray(self.vector))
                if len(temp_vector.shape) > 1:
                    raise TypeError("Parameter vector must be a 1-dimensional array-like")
    
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
    
    def __add__(self, other):
        return self.fitness_val + other.fitness_val
    
    def __sub__(self, other):
        return self.fitness_val - other.fitness_val
    
    def __abs__(self):
        return abs(self.fitness_val)
    
    def mate(self, other):
        """Mating Function
        
        Arguments:
            other {Genetic_Vector} -- The other parent
        
        Raises:
            TypeError: If other is not a Genetic Vector
            ValueError: If other is an incompatible Genetic Vector
        
        Returns:
            Genetic_Vector -- Child vector
        """

        if not isinstance(other, Genetic_Vector):
            raise TypeError("Other parent is invalid")
        
        if self.template.vector_length != other.vector_length:
            raise ValueError("Vector lengths do not match")
        
        #Initialize the child vector
        child_vector = np.empty(self.template.vector_length)
        
        #If we are doing simple crossover, we're just going to take one of the two parent parameters
        if self.template.simple_crossover:
            for i in range(self.template.vector_length):
                parents = [self, other]
                rand_parent = random.randint(0,1)
                child_vector[i] = parents[rand_parent].vector[i]
        #If we are not doing simple crossover, we're going to evaluate the complex crossover
        else:
            for i in range(self.template.vector_length):
                # parent_param_tuple = (self.vector[i], other.vector[i])
                # param_bounds = self.template.parameter_bounds[i]
                child_vector[i] = self._complex_crossover((self.vector[i], other.vector[i]), self.template.parameter_bounds[i])
        
        if self.template.mutation:
            child_mutation_rate = self.template.mutation_rate + (random.random() * (0.4) - 0.2) * self.template.mutation_rate
            child_vector = self.mutate(child_vector, child_mutation_rate)

        #Set child's generation number to be one more than the greater of either parent's
        child_generation = max(self.generation, other.generation) + 1
        
        #Create the genetic vector
        child = Genetic_Vector(child_vector, child_generation, self.template, False)

        return child

    def mutate(self):
        """Mutation Function
        
        Returns:
            np.array() -- Mutated vector
        """
        r = np.random.sample(self.template.vector_length)
        v = np.copy(self.vector)
        for i,x in enumerate(r):
            if x < self.template.mutation_rate:
                counter = 0
                k = np.random.normal(v[i], .005)
                while k > self.template.parameter_bounds[i][1] or k < self.template.parameter_bounds[i][0]:
                    counter += 1
                    k = np.random.normal(v[i], .005 ** counter)
                v[i] = k
        
        return v

        
    
    def _complex_crossover(self, params, specific_bounds):
        """Custom crossover function
        
        Arguments:
            params {2-tuple} -- The appropriate parameter for both parents
            specific_bounds {2-tuple} -- Upper and lower bounds for param
        
        Returns:
            float -- Result of the crossover function
        """

        candidate_value = self.template.crossover_function(params[0], params[1])
        if len(specific_bounds) == 3:
            exceptions = specific_bounds[-1]
        else:
            exceptions = []
        #Loop so long as the candidate is not within the bounds 
        while candidate_value < specific_bounds[0] or candidate_value > specific_bounds[1] or candidate_value in exceptions:
            candidate_value = self.template.crossover_function(params[0], params[1])
        
        return candidate_value

class Genetic_Pool:
    def __init__(self, template, size = None, elitism = 0, selection_method = "tournament"):
        """Create a Genetic Pool object
        
        Arguments:
            template {Genetic_Template} -- Genetic Template the pools will use
        
        Keyword Arguments:
            size {int} -- Size of the pool. Will default to ten times the number of hyperparameters to optimize (default: {None})
            elitism {int} -- How many individuals will be held over to the next generation (default: {0})
            selection_method {str} -- Parent selection method (Currently only supports tournament selection) (default: {"tournament"})
        
        Raises:
            TypeError: Raised if template is not a Genetic_Template object
        """
        if not isinstance(template, Genetic_Template):
            raise TypeError("Parameter template must be a Genetic_Template object")

        self.template = template
        self.elitism = elitism
        self.members = (self.template.vector_length * 10) if not size else size
        self.dimensions = self.template.vector_length
        self.selection_method = selection_method
        self.pool = []
        self.fxn_dict = {
                "tournament": self._tournament
            }
    
    def _parent_selection(self):

        try:
            optimization_chosen = self.fxn_dict[self.selection_method.lower()]
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
            candidate_indices = frandom.sample(np.arange(self.members, dtype=np.int64), tournament_size).tolist()
        except ValueError:
            i, j = frandom.sample(np.arange(self.members, dtype=np.int64), 2)
            return [self.pool[i], self.pool[j]]
        
        rounds = (tournament_size - 1).bit_length() - 1

        for i in range(rounds):
            next_round = []
            while len(candidate_indices) > 1:
                next_round.append(candidate_indices[0] if self.pool[candidate_indices[0]] > self.pool[candidate_indices[1]] else candidate_indices[1])
                candidate_indices = candidate_indices[2:]

            candidate_indices += next_round
        
        return [self.pool[candidate_indices[0]], self.pool[candidate_indices[1]]]

    
    def new_generation(self, nelder_mead_elitism = None):
        """Creates a new generation

        Keyword Arguments:
            nelder_mead_elitism {int} -- 
        
        Raises:
            RuntimeError: self.initialize() must be ran at least once before using this function
        
        Returns:
            Genetic_Vector[] -- List of Genetic Vectors
        """
        nelder_mead_elitism = nelder_mead_elitism if nelder_mead_elitism is not None else True
        gen_size = self.members
        next_gen = []
        if len(self.pool) == 0:
            raise RuntimeError("New Generation cannot be generated without first initializing using the initialize() method")
        
        if self.elitism:
            #Primary Elitism: Send the best candidate to the next generation unaltered
            next_gen.append(self.pool[-1])
            
            for i in range(self.elitism - 1):
                next_gen.append(Genetic_Vector(self.pool[-1].mutate(), self.pool[0].generation + 1, self.template, type_check=False))
            
        
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
        
        if nelder_mead_elitism:
            self.pool[0] = nelder_mead(self.pool[- (self.dimensions + 1):])
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
                    
                    if isinstance(lower, float) or isinstance(upper, float):
                        value = random.random() * distance + lower
                    else:
                        value = random.randint(lower, upper)
                    
                    vector_list.append(value)
                new_genetic_vector = Genetic_Vector(np.array(vector_list), 1, self.template, False)
                return new_genetic_vector
        
        for i in range(self.members):
            self.pool.append(generation_function())
        
        self.pool.sort()
        self.generation_max = 0
        self.score_max = self.pool[-1].fitness_val
    
    def initialize(self, generation_function = None):
        self.initialization(generation_function)
    
    def multiple_generations(self, generations, early_stopping = True, stopping_percentage = 0.3, nelder_mead_elitism = None):
        """Runs multiple generations
        
        Arguments:
            generations {int} -- Number of generations you want to run
        
        Keyword Arguments:
            early_stopping {bool} -- Whether or not to use early stopping (default: {True})
            stopping_percentage {float} -- Early Stopping Percent of Generations. If early_stopping is true, how much of the generation without changes should indicate stopping (default: {0.3})
            nelder_mead_elitism {int} -- Determines whether or not to use nelder-mead elitism
        
        Returns:
            Genetic_Vector -- Returns the strongest genetic vector in the pool
        """
        nelder_mead_elitism = nelder_mead_elitism if nelder_mead_elitism is not None else True
        for i in range(generations):
            self.new_generation(nelder_mead_elitism)
            if self.pool[-1].fitness_val > self.score_max * 1.01:
                self.score_max = self.pool[-1].fitness_val
                self.generation_max = i + 1
            elif early_stopping:
                if (i - self.generation_max) > (stopping_percentage * generations):
                    print(f"Stopped at generation {i}")
                    break
        
        return self.pool[-1]
    
    def plot(self, res = .01):
        """Plots the fitness function and where each member of the pool lies on the generated surface
        
        Keyword Arguments:
            res {float} -- Resolution. If greater than 1, will represent how many entries per hyperparameter. If less than 1, will represent the percent step (default: {.01})
        
        Raises:
            ValueError: Feature is only available for 2 hyperparameters
        """
        if self.template.vector_length != 2:
            raise ValueError("Feature only available for 2 hyperparameters")
        
        if res > 1:
            res = 1 / res
        
        #Initialize plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        #Set up X and Y
        x_bounds = self.template.parameter_bounds[0]
        X = np.arange(x_bounds[0], x_bounds[1], res * (x_bounds[1] - x_bounds[0]))
        y_bounds = self.template.parameter_bounds[1]
        Y = np.arange(y_bounds[0], y_bounds[1], res * (y_bounds[1] - y_bounds[0]))
        X, Y = np.meshgrid(X, Y)

        #Find total length for progress report
        total_length, count = len(X) * len(Y), 0
        zrows = []
        #Evaluate Z at every point on the grid, (X,Y)
        for i,x in enumerate(X):
            zrow = []
            for j,y in enumerate(Y):
                zrow.append(self.template.fitness(x[i], y[j]))
                count += 1
                if count % 10 == 1:
                    print("{0: .1%}".format(count / total_length), end='\r')
            zrows.append(zrow)
        Z = np.array(zrows)

        #Plot surface X, Y and Z
        ax.plot_surface(X, Y, Z)
        xvals, yvals, zvals = [], [], []

        #Plot 3 best differently
        for i in self.pool[:-3]:
            x, y = i.vector
            z = i.fitness_val + .01
            xvals.append(x)
            yvals.append(y)
            zvals.append(z)

        best_xvals, best_yvals, best_zvals = [], [], []
        for i in self.pool[-3:]:
            x, y = i.vector
            z = i.fitness_val + .01
            best_xvals.append(x)
            best_yvals.append(y)
            best_zvals.append(z)
        
        #Plot fitness of every Genetic Vector in the pool
        ax.plot(xvals, yvals, zvals, 'ro')
        ax.plot(best_xvals, best_yvals, best_zvals, 'bo')
        plt.show()

class Genetic_Archipelago:
    def __init__(self, pool_templates, size = 2):
        """Creates a Genetic_Archipelago to optimize several pools at once. Allows comparison between multiple fitness functions
        
        Arguments:
            pool_templates {Genetic_Template[]} -- List of Genetic Templates to be used for the pools
        
        Keyword Arguments:
            size {int} -- Number of islands each genetic template will create (default: {2})
        """
        self.islands = size
        if not isinstance(pool_templates, list):
            self.pool_templates = [pool_templates]
        else:
            self.pool_templates = pool_templates
    
    def initialize(self):
        self.pools = []

        for i in range(self.islands):
            for j in self.pool_templates:
                k = Genetic_Pool(j.template, j.members, j.elitism, j.selection_method)
                k.initialization()
                self.pools.append(k)
        
    def multiple_generations(self, generations, early_stopping = True, stopping_percentage = 0.3):
        k = []
        for i, x in enumerate(self.pools):
            k.append([x.multiple_generations(generations, early_stopping, stopping_percentage), i])
        
        k.sort()
        return self.pools[k[-1][1]]
    
    def hybrid_nm(self, generations):
        """A hybrid method of genetic algorithm with nelder_mead. STILL IN DEVELOPMENT. When completed, should combine the robust search power of genetic algorithms with the local optimization of Nelder-Mead
        
        Arguments:
            generations {int} -- Number of generations to run
        
        Returns:
            Genetic_Vector -- Best genetic vector of pool obtained from archipelago search with local maximization applied
        """
        k = []
        for x in self.pools:
            k.append([x.multiple_generations(generations), x])
        
        k.sort(key=lambda x: x[0])
        print(k[-1][0].fitness_val)
        #NOTE: Will grab top n Genetic Vectors from the best pool
        best_pool = k[-1][1]
        nm_members = best_pool.pool[-(int(best_pool.members//10) + 1):]
        return nelder_mead(nm_members)



    def select_best_pool(self, step_generations = 5, nelder_mead_elitism = True, early_stopping = True):
        """Selects the best pool out of the given pools
        
        Keyword Arguments:
            step_generations {int} -- Number of generations before nuking an island (default: {5})
            nelder_mead_elitism {bool} -- Refers to whether it will apply Nelder-Mead after each step (default: {True})
        
        Returns:
            Genetic_Pool -- The strongest genetic pool
        """
        temp_list = []
        for index, pool in enumerate(self.pools):
            temp_list.append([pool.multiple_generations(step_generations, early_stopping = early_stopping), index])
        
        temp_list.sort()
        temp_list = temp_list[int(len(temp_list)//2):]
        while len(temp_list) > 1:
            for pool, index in temp_list:
                pool = self.pools[index].multiple_generations(step_generations, early_stopping = early_stopping)
            temp_list.sort()
            temp_list=temp_list[int(len(temp_list)//2):]
        return self.pools[temp_list[0][1]]
    
    def optimize(self, step_generations = 5, after_generations = 10, nelder_mead_elitism = True, early_stopping = False):
        best_pool = self.select_best_pool(step_generations=step_generations, nelder_mead_elitism=nelder_mead_elitism, early_stopping=early_stopping)
        return best_pool.multiple_generations(after_generations, nelder_mead_elitism=nelder_mead_elitism, early_stopping = early_stopping)


def nelder_mead(arg, max_shrinks = 3, max_count = None, epsilon = 1e-2):
    """Applies Nelder-Mead simplex method to locally optimize between multiple Genetic_Vectors
    
    Arguments:
        arg {Genetic_Vector[]} -- List of Genetic_Vectors the simplex will use
    
    Keyword Arguments:
        max_shrinks {int} -- Maximum number of allowed shrinks in a row before it breaks (default: {3})
        max_count {int or None} -- Default forces max_count to be 20 times the number of parameters being optimized (default: {None})
        epsilon {float} -- Tolerance for optimization. Currently unused (default: {1e-2})
    
    Returns:
        Genetic_Vector -- Locally optimized Genetic_Vector
    """


    #Create a shallow copy of the list of Genetic Vectors
    li = arg[:]
    li.sort()
    #Find the centroid of the list
    centroid = np.mean([x.vector for x in li[1:]], axis=0)
    shrank_in_a_row, count = 0, 0
    n = len(li[1:])
    max_count = max_count if max_count else 20 * (n + 1) 

    #Iterate until it shrinks 5 times in a row
    while(shrank_in_a_row < max_shrinks):
        #Find worst, next worst, and best genetic vectors
        worst, second_worst, best = *li[:2], li[-1]
        shrank = False
        #Create a new genetic vector we will attempt to check if it's accurate. NOTE: PENALTY CHECK SHOULD BE ENABLED TO ALLOW REJECTION OF VECTORS THAT GO OUT OF BOUNDS
        new_try = Genetic_Vector(2 * centroid - worst.vector, worst.generation + 1, worst.template, type_check= False, penalty_check = True)
        #Check if the new candidate is better than the second worst genetic vector
        if new_try >= second_worst:
            #Set the input to the new candidate
            inp = new_try
            #If the new candidate is better than the best genetic vector, we can skip an iteration and try to create an even better candidate
            if new_try >= best:
                newer_try = Genetic_Vector(2 * new_try.vector - centroid, worst.generation + 1, worst.template, type_check=False, penalty_check = True)
                #If this better candidate is also better than the best, this will be our input
                if newer_try >= best:
                    inp = newer_try

        #If the new candidate is worse than our second worse genetic vector            
        else:
            #But it's better than our worst genetic vector
            if new_try >= worst:
                #Create a newer candidate between the candidate and the centroid
                newer_try = Genetic_Vector((new_try.vector + centroid) / 2, worst.generation + 1, worst.template, type_check=False, penalty_check = True)
            #But if it's worse than even our worst genetic vector
            else:
                #Create a newer candidate between the worst vector and the centroid
                newer_try = Genetic_Vector((worst.vector + centroid) / 2, worst.generation + 1, worst.template, type_check=False, penalty_check = True)
            
            #If the newer candidate is better than both the new candidate and the worst vector, set the input to the new candidate
            if newer_try > max(worst, new_try):
                inp = newer_try
            else:
                #Otherwise, shrink the list of genetic vectors and duplicate the worst one
                for j in range(n):
                    li[j] = Genetic_Vector((li[j].vector + best.vector)/2, worst.generation + 1, worst.template, type_check=False, penalty_check = True)
                inp = li[0]
                shrank = True

        #Replace the worst vector with the input
        li[0] = inp
        #Sort the list
        li.sort()
        #Recompute centroid. Shrinking will take considerably more resources than not shrinking.
        if not shrank:
            if inp > second_worst:
                diff = (1 / n) * (second_worst.vector - inp.vector)
                centroid -= diff
            if shrank_in_a_row:
                shrank_in_a_row -= 1
        else:
            centroid = np.mean([x.vector for x in li[1:]], axis=0)
            shrank_in_a_row += 1
        
        count += 1
        if count > max_count:
            break
    return li[-1]