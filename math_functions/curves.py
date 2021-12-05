from typing import Union, Optional
import numpy as np

class BezierClass:
    def __init__(self,  degree:Optional[int] = None, 
                        control_points: Optional[list[list[float]]] = None,
                        coefficient_matrix: Optional[np.ndarray] = None) -> None:
        """Creates a bezier curve of the input degree and control points.
        Args:
            degree (int): The degree of the bezier curve to be used
            control_points (list[float]): The points that define the curve
        """
        #Check if the inputs are all empty or all provided
        if degree is None and control_points is None and coefficient_matrix is None:
            raise ValueError("Inputs must contain control points and either the bezier curve degree or the coefficient matrix")
        elif degree is not None and control_points is not None and coefficient_matrix is not None:
            raise ValueError("Inputs must contain control points and either the bezier curve degree or the coefficient matrix")
        elif control_points is None:
            raise ValueError("Inputs must contain control points")
        

        self.binomial           = BinomialClass()
        self.control_points     = np.asarray(control_points)
        self.lookup_table       = None

        #If the degree is provided, create the coefficient matrix
        if coefficient_matrix is None:
            self.binomial.precompute(degree)
            self.coefficient_matrix = self.generate_coefficient_matrix(degree, control_points)
            self.degree             = degree
        else:
            #Otherwise, calculate the degree
            matshape = coefficient_matrix.shape
            if matshape[0] != matshape[1]:
                raise ValueError("Coefficient matrix must be square")
            self.degree = matshape[0] - 1
            self.coefficient_matrix = coefficient_matrix
    
    def generate_coefficient_matrix(self, degree:int, control_points: list[list[float]]) -> np.ndarray:
        """Creates a bezier curve of the input degree and control points.
        Args:
            degree (int): The degree of the bezier curve to be used
            control_points (list[float]): The points that define the curve
        """
        if len(control_points) != (degree + 1):
            raise ValueError(f"Bezier curve of degree {degree} requires {degree + 1} control points")
        
        coefficient_matrix = []
        for i in range(degree+1):
            j = degree - i
            binomial_row = self.binomial.binomial_negative(j)
            #Append zeros to the beginning of the row to make it of length degree + 1
            binomial_row        = [0]*(degree+1-len(binomial_row)) + binomial_row
            multiplier          = self.binomial.binomial(degree, i)
            coefficient_matrix += [[multiplier * weight for weight in binomial_row]]
        
        return np.asarray(coefficient_matrix).T
    
    def compute_lookup_table(self, num_samples:int = 50) -> None:
        """Computes a lookup table relating arc distance to the curve's t values
        Args:
            num_samples (int): Number of samples to use. Greater number of samples will indicate a more accurate curve, but
            will also take longer to compute. Defaults to 50.
        """
        #Generate a list of points on the curve
        curve_points = self.sample_points(num_samples)
        
        #Compute the distance between each point on the curve
        curve_distances = [0]
        for i in range(1, len(curve_points)):
            curve_distances +=[np.linalg.norm(curve_points[i] - curve_points[i-1], ord=2)]
        
        #Compute the cumulative distance
        curve_distances = np.cumsum(curve_distances)
        
        #Normalize the curve distances
        curve_distances /= curve_distances[-1]
        
        #Create the lookup table
        lookup_table = []
        for i in range(len(curve_distances)):
            lookup_table += [[curve_distances[i], i/(len(curve_distances) - 1)]]
        
        self.lookup_table = np.array(lookup_table)
        return
    
    def sample_points(self, weights:Union[list[float], tuple[float], int]) -> list[float]:
        """Uses a bezier curve of the input degree to generate a sample of points based on the weights.
        
        Args:
            weights (list[float], tuple[float], int): The weights used to generate the sample of points. If fed an integer,
            will generates a linear number of points
        
        Returns:
            list[float]: A list of points on the bezier curve, including endpoints
        """
        if isinstance(weights, int):
            weights = np.linspace(0, 1, weights, endpoint=True, dtype=float)
        
        #Turn the weights vector into a matrix of form t^i where i is the column index
        weights = np.asarray([[w ** i for i in range(self.degree + 1)] for w in weights])
        
        
        #Multiply the coefficient matrix by the weights matrix
        return weights @ self.coefficient_matrix @ self.control_points
    
    def sample_points_evenly_spaced(self, num_samples:int) -> list[float]:
        """Uses the lookup table to generate evenly spaced points on the curve, interpolating if necessary.
        
        Args:
            num_samples (int): The number of points to generate
        
        Returns:
            list[float]: A list of points on the bezier curve, including endpoints
        """
        #Check if the lookup table has been computed
        if self.lookup_table is None:
            self.compute_lookup_table()
        
        #Generate evenly spaced points
        evenly_spaced_points = np.linspace(0, 1, num_samples, endpoint=True, dtype=float)
        
        #Using the lookup table, find the t values that correspond to the evenly spaced points and interpolate if necessary
        evenly_spaced_t_values = []
        for i in range(num_samples):
            #Find the closest t value in the lookup table
            current_point = evenly_spaced_points[i]
            evenly_spaced_t_values += [self.lookup_nearest_t_value(current_point)]
        
        #Generate the points using the generated t values, evenly spaced by arc length
        return self.sample_points(evenly_spaced_t_values)

    def lookup_nearest_t_value(self, arc_distance:float) -> float:
        """Finds the t value that corresponds to the input arc distance.
        
        Args:
            arc_distance (float): The arc distance to find the t value for
        
        Returns:
            float: The t value that corresponds to the input arc distance
        """
        #Check if the lookup table has been computed
        if self.lookup_table is None:
            self.compute_lookup_table()
        
        #Calculate all the t values in the lookup table that are less than or equal to the input arc distance, then return the last one
        closest_t_value = self.lookup_table[self.lookup_table[:, 0] <= arc_distance][-1]
        #If the closest t value is the same as the input arc distance, return the t value
        if closest_t_value[0] == arc_distance:
            return closest_t_value[1]
        else:
            #If the closest t value is not the exact value, interpolate
            where_row                       = np.where(self.lookup_table[:, 0] == closest_t_value[0])[0][0]
            lower_t_value, upper_t_value    = self.lookup_table[where_row: where_row + 2]
            #Rename variables for readability
            lower_arc_distance, upper_arc_distance  = lower_t_value[0], upper_t_value[0]
            lower_t_value, upper_t_value            = lower_t_value[1], upper_t_value[1]
            #Interpolate
            t_value = lower_t_value + (arc_distance - lower_arc_distance) * (upper_t_value - lower_t_value) / (upper_arc_distance - lower_arc_distance)
            return t_value
    
    def derivative(self, order:int = 1) -> 'BezierClass':
        """Generate the bezier curve from the derivative
        Args:
            order (int, optional): The nth derivative. Defaults to 1.
        Returns:
            BezierClass: The corresponding curve
        """
        #If the order is 0, return the original curve
        if order == 0:
            return self
        #Generate the derivative coefficient matrix
        derivative_matrix = []
        for i in range(self.coefficient_matrix.shape[0]):
            derivative_matrix += [i * self.coefficient_matrix[:, i]]
        derivative_matrix = np.asarray(derivative_matrix).T[1:, 1:]

        #Generate the derivative control points
        derivative_control_points = np.diff(self.control_points, axis=0)

        #Create the derivative curve and recurse to the given order
        return BezierClass(coefficient_matrix=derivative_matrix, control_points=derivative_control_points).derivative(order - 1)
    
    def curvature(self, t:float) -> float:
        """Computes the curvature of the curve.

        Args:
            t (float): The t value to compute the curvature at
        
        Returns:
            float: The curvature of the curve
        """
        #Evaluate the vector of the curve at the t value
        derivative_curve        = self.derivative()
        second_derivative_curve = derivative_curve.derivative()
        derivative_point        = derivative_curve.evaluate(t)
        second_derivative_point = second_derivative_curve.evaluate(t)

        #Calculate the norm of the first derivative point as the denominator
        norm_derivative_point = np.linalg.norm(derivative_point)

        #Calculate the wedge product of the vectors for the first and second derivatives using einsum
        wedge_product = np.einsum('i,j', derivative_point, second_derivative_point)

    
    def evaluate(self, t:float) -> float:
        """Evaluate the point on the Bezier curve at t
        Args:
            t (float): t value to evaluate
        Returns:
            float: The value of the point on the curve at t
        """
        return self.sample_points([t])[0]

class BinomialClass():
    def __init__(self):
        self.max_computed = 0
        self.binomial_cache = [{0:1}]
    
    def __repr__(self) -> str:
        return f"BinomialClass(max degree: {self.max_computed})"
    
    def binomial_method(func):
        """Wrapper for all methods in the binomial class, checks if n exceeds the maximum computed level"""
        def wrapper(*args, **kwargs):
            self = args[0]
            n = kwargs.get("n", args[1])
            if n > self.max_computed:
                self.precompute(n)
            return func(*args, **kwargs)
        return wrapper
    
    def precompute(self, max_computed: int):
        """Precomputes the binomial coefficients up to the input max_computed.
        
        Args:
            max_computed (int): The maximum value to precompute
        """
        if self.max_computed > max_computed:
            return

        for i in range(self.max_computed + 1, max_computed+1):
            self.binomial_cache += [{}]
            for j in range(i+1):
                self.binomial_cache[i][j] = self.binomial_cache[i-1].get(j-1, 0) + self.binomial_cache[i-1].get(j, 0)

        self.max_computed = max_computed
    
    @binomial_method
    def binomial(self, n: int, k: int) -> int:
        """Returns the binomial coefficient of n and k.
        
        Args:
            n (int): The value of n
            k (int): The value of k
        
        Returns:
            int: The binomial coefficient of n and k
        """
        return self.binomial_cache[n][k]
    
    @binomial_method
    def binomial_level(self, n: int) -> list[int]:
        """Generates the binomial coefficients of n.
        
        Args:
            n (int): The value of n
        
        Returns:
            list[int]: The binomial coefficients of n
        """
        return [
            value
            for key, value 
            in self.binomial_cache[n].items()
            ]
    
    @binomial_method
    def binomial_negative(self, n: int) -> list[int]:
        """Returns the coefficient of the binomial expansion of (1-x)^n.
        Args:
            n (int): The value of n
        
        Returns:
            list[int]: The binomial coefficients of (1-x)^n
        """
        binomial_level = self.binomial_level(n)
        #Alternate the sign of the binomial coefficients between positive and negative
        return [
            value * (-1)**(i % 2)
            for i, value
            in enumerate(binomial_level)
            ]