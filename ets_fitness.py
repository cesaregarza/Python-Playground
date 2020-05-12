import data_analysis_functions_redux as das
import genetic
from numba import jit, double, boolean
import numpy as np
import pandas as pd
import random
from scipy.optimize import broyden1
from scipy.special import erf


#Crossover with a ~4.5% chance of mutation
@jit(double(double, double), nopython=True)
def float_crossover(mom, dad):
    avg = (mom + dad) / 2
    std = abs(mom - dad) / 4
    return np.random.normal(avg, std)

#Crossover with arbitrary chance of mutation. Please note that this will return a function, do not feed it directly.
def float_crossover_mutation(mutation_chance):

    #Function to optimize
    def F(x):
        return erf(x/np.sqrt(2)) + mutation_chance - 1
    
    #Find n
    n = broyden1(F, [1.0])[0]

    #Create the crossover function
    @jit(double(double, double), nopython=True)
    def crossover(mom, dad):
        avg = (mom + dad) / 2
        std = abs(mom - dad) / (2 * n)
        return np.random.normal(avg, std)
    
    return crossover


class ets:
    def __init__(self, df, max_date = pd.Timestamp.today(), elitism=1, optimizer="genetic hybrid", size=2, mutation_chance=0.045):
        
        if isinstance(max_date, int):
            max_date = df.index[-1]

        if isinstance(df, pd.Series):
            date_range = pd.date_range(start=df.index[0], end=max_date,freq='M')
            df = df.reindex(date_range)
            li = df.to_numpy() + 0.0
            li = np.nan_to_num(li)
        else:
            raise TypeError("Input should be a Series")
        
        self.li = li
        self.df = df
        self.size = size
        self.seasonal = False
        self.lag = 0

        # seasonal_lag = self._find_seasonality(self.df)
        # if abs(seasonal_lag[0]) > 0.8:
        #     self.seasonal = True
        #     self.lag = seasonal_lag[1]

        if mutation_chance != 0.045:
            crossover = float_crossover_mutation(mutation_chance)
        else:
            crossover = float_crossover

        min_val, max_val = np.min(self.li), np.max(self.li)
        diff_vals = max_val - min_val
        alpha_bounds = (0.02, 1.0)
        param_bounds = (0.0, 1.0)
        level_bounds = (0, max_val * 1.5)
        trend_bounds = (-diff_vals, diff_vals)
        level_bounds_m = (0, max_val * 1.5, [0])
        trend_bounds_m = (-diff_vals, diff_vals, [0])
        # s_list = [trend_bounds] * self.lag
#                                             alpha         beta            initial_level   initial_trend  phi/gamma   s_list       fitness fxn             mutation          simple crossover          crossover function
        template1 = genetic.Genetic_Template([alpha_bounds,                 level_bounds],                                          self.fitness_aia,       mutation = False, simple_crossover = False, crossover_function=crossover)
        template2 = genetic.Genetic_Template([alpha_bounds, param_bounds,   level_bounds,   trend_bounds],                          self.fitness_abiaib,    mutation = False, simple_crossover = False, crossover_function=crossover)
        template3 = genetic.Genetic_Template([alpha_bounds, param_bounds,   level_bounds,   trend_bounds, param_bounds],            self.fitness_abiaibd,   mutation = False, simple_crossover = False, crossover_function=crossover)
        template4 = genetic.Genetic_Template([alpha_bounds, param_bounds,   level_bounds_m, trend_bounds_m],                        self.fitness_abiaib_m,  mutation = False, simple_crossover = False, crossover_function=crossover)
        # template5 = genetic.Genetic_Template([alpha_bounds, param_bounds,   level_bounds,   trend_bounds, param_bounds, *s_list],   self.fitness_abiaibgs,  mutation = False, simple_crossover = False, crossover_function=crossover)

        pool1 = genetic.Genetic_Pool(template1, elitism=elitism)
        pool2 = genetic.Genetic_Pool(template2, elitism=elitism)
        pool3 = genetic.Genetic_Pool(template3, elitism=elitism)
        pool4 = genetic.Genetic_Pool(template4, elitism=elitism)
        # pool5 = genetic.Genetic_Pool(template5, elitism=elitism)

        pools = [pool1, pool2, pool3]
        if not len(li[li == 0]) and len(li) > 1:
            pools.append(pool4)
            pass
        
        # if seasonal:
        #     pools.append(pool5)

        self.arch = genetic.Genetic_Archipelago(pools, size=size)
        self.arch.initialize()
    
    def optimize(self, step_generations = 5, after_generations = 10, nelder_mead_elitism = True, verbose = False, print_vector = False):
        if len(self.li) <= 3:
            return [self.df[-1]] * 3
        best_vector = self.arch.optimize(step_generations = step_generations, after_generations=after_generations, nelder_mead_elitism=nelder_mead_elitism, early_stopping=False)
        
        vec = best_vector.vector
        if print_vector:
            print(vec)

        alpha = vec[0]
        if len(vec) == 2:
            initial_level = vec[1]
            beta, initial_trend, phi = 0,0,0
        else:
            beta = vec[1]
            initial_level = vec[2]
            initial_trend = vec[3]
            phi = vec[4] if len(vec) == 5 else 0
        try:
            return das.exponential_smoothing(self.df, alpha, beta, initial_level=initial_level, initial_trend = initial_trend, phi=phi, verbose=verbose)
        except ValueError:
            try:
                return [self.df[0]] * 3
            except IndexError:
                return [0, 0, 0]



    def fitness_aia(self, alpha, initial_level):
        return das._fast_ets_aia(self.li, alpha, initial_level)
    
    def fitness_abiaib(self, alpha, beta, initial_level, initial_trend):
        return das._fast_ets_abiaib(self.li, alpha, beta, initial_level, initial_trend)
    
    def fitness_abiaibd(self, alpha, beta, initial_level, initial_trend, phi):
        return das._fast_ets_abiaibd(self.li, alpha, beta, initial_level, initial_trend, phi)
    
    def fitness_abiaib_m(self, alpha, beta, initial_level, initial_trend):
        return das._fast_ets_abiaib_m(self.li, alpha, beta, initial_level, initial_trend)
    
    # def fitness_abiaibgs(self, alpha, beta, initial_level, initial_trend, gamma, *args):
    #     return das._fast_ets_abiaibgs(self.li, alpha, beta, initial_level, initial_trend, gamma, np.asarray[*args])
    
    def generation_function_m(self, template):
        vector_list = []
        for lower, upper in template.parameter_bounds:
            distance = upper - lower
            
            if isinstance(lower, float) or isinstance(upper, float):
                value = random.random() * distance + lower
                while value == 0:
                    value = random.random() * distance + lower
            else:
                value = random.randint(lower, upper)
                while value == 0:
                    value = random.randint(lower, upper)

            
            vector_list.append(value)
        new_genetic_vector = genetic.Genetic_Vector(np.array(vector_list), 1, template, False)
        return new_genetic_vector
    
    # def _find_seasonality(self, series, min_lag = 4, max_len = 12):
    #     high_val, lag = 0, 0
    #     for i in range(max_len + 1 - min_lag):
    #         val = series.autocorr(lag=(i + min_lag))
    #         if abs(val) > high_val:
    #             high_val = val
    #             lag = i + min_lag
        
    #     return [high_val, lag]
    
    
    
