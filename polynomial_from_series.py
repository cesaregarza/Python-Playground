# %%
from numpy.polynomial.polynomial import Polynomial
from sympy import nsimplify, symbols, factor
from sympy import Poly as SymPoly
import numpy as np


# %%
class Series:
    def __init__(self, array, series_type = "series"):
        self._polynomial =  []
        self._series =      []

        if series_type == "polynomial":
            self._polynomial =  Polynomial(array)
        elif series_type == "series":
            self._series =      np.array(array)
        else:
            raise TypeError("Invalid Type")
    
    def find_degree(self, array, degree=1, max_degree = 50):

        if (len(array) == 1) or (degree == max_degree):
            return False
        
        #Find the list of differences
        diff_list = np.diff(array)
        
        #Stop if all numbers in the difference list are close enough to equal, otherwise recurse
        if np.allclose(diff_list, [diff_list[0] for x in diff_list]):
            return [diff_list[0], degree]
        else:
            return self.find_degree(diff_list, degree+1)
    
    def generate_factorials(self, n):
        factorial_array = [1]
        factorial_array.extend([None] * n)
        for i in range(n):
            factorial_array[i + 1] = factorial_array[i] * (i + 1)
        
        return factorial_array
    
    def find_polynomial(self, max_degree = 50, max_loops = 99, return_type = "sympy", polynomial_variable = "x"):
        polynomial_array =  []
        factorials_list =   self.generate_factorials(max_degree)
        working_array =     np.copy(self._series)

        loop_count = 0

        current_degree = 999

        while current_degree > 0:
            #Increment loop count and raise recursion error if there's too many loops
            loop_count += 1
            if loop_count > max_loops:
                raise RecursionError(f"Could not determine polynomial in given maximum {max_loops} loops. Try again with a higher loop count")
            
            #Find the preliminary coefficient and the degree through iterative differences
            coefficient, temporary_degree = self.find_degree(working_array)
            
            #Ending condition since find_degree will return a minimum of degree 1
            if temporary_degree == current_degree:
                temporary_degree -= 1
                coefficient = working_array[0]

            #Set current_degree and correct the coefficient by dividing by the appropriate factorial
            current_degree =    temporary_degree
            coefficient /=      factorials_list[current_degree]
            
            #If the polynomial array is empty, start. Otherwise, slot the coefficient in the appropriate space
            if len(polynomial_array) == 0:
                polynomial_array =      np.zeros(current_degree + 1)
                polynomial_array[-1] =  coefficient
            else:
                polynomial_array[current_degree] = coefficient
            
            #Use the polynomial we're working with to generate a series of the same length of the input series
            generated_polynomial_array = np.array([Polynomial(polynomial_array)(x) for x in range(len(working_array))])
            
            #Replace our working array with the difference between the generated series and the input series.
            working_array = self._series - generated_polynomial_array
        
        self._polynomial = Polynomial(polynomial_array)

        if return_type == "numpy":
            return self._polynomial
        elif return_type == "sympy":
            x = symbols(polynomial_variable)
            return nsimplify(SymPoly(self._polynomial.coef[::-1], x))
        else:
            raise ValueError(f"Unknown return type {return_type}")
            


# %%
a = np.cumsum([x ** 3 + 5 * x ** 2 - 8 * x + 3 for x in range(10)])
b = Series(a)
c = b.find_polynomial(polynomial_variable="n")
factor(c)


# %%
