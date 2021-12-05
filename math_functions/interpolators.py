from abc import ABC, abstractmethod
from typing import Optional
from .curves import BezierClass
import numpy as np
from sklearn.linear_model import LinearRegression

class InterpolatorError(Exception):
    pass

class InterpolatorMixin(ABC):
    @abstractmethod
    def __init__(self, control_points: np.ndarray) -> None:
        "Initialize the interpolator"
    
    @abstractmethod
    def fitted_method(func):
        "Wrapper for functions that require the interpolator to be fitted prior to use"
    
    @abstractmethod
    def fit(self, control_points: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        "Fit the interpolator. The variable names were chosen to also fit parametrized models"
    
    @abstractmethod
    def is_fitted(self) -> bool:
        "Check if the interpolator is fitted"

    @abstractmethod
    def interpolate(self, points:np.ndarray) -> np.ndarray:
        "Uses the input points to interpolate"
    
    @abstractmethod
    def sample(self, num_samples:int) -> np.ndarray:
        "Generates evenly spaced points from the smallest control point to the largest one based on the number of input samples"
    
    @abstractmethod
    def get_value(self, t:float) -> float:
        "Given the input point, return the interpolator's value at that point"
    
    @abstractmethod
    def get_derivative(self, t:float, order:Optional[int] = 1) -> float:
        "Return the nth order derivative of the interpolation model at the given point"

class nDegreeBezierInterpolator(InterpolatorMixin):
    """ Generates a (n-1)-degree Bezier curve from the given control points for interpolation by arc length.
        PROS:
            - Easy to understand
            - Outputs a guaranteed smooth curve
            - Can be used for interpolation of any n-dimensional curve
            - When control points are close together, the curve will be close to the control points
        CONS:
            - Does not scale well with the number of points
            - Highly unlikely to pass directly through the control points
            - Accuracy relies on the number of samples of the arc length being high enough
    """

    def __init__(self, arc_length_samples:int = 200) -> None:
        """Generates a (n-1)-degree Bezier curve from the given control points for interpolation.
        Args:
            arc_length_samples (int): The number of samples to use for calculating the lookup table relating arc length to t-values.
        """
        self.degree             = None
        self.bezier_curve       = None
        self.control_points     = None
        self.arc_length_samples = arc_length_samples
        self.fitted             = False
    
    def fitted_method(func):
        """Decorator for methods that require the interpolator to be fitted.
        Args:
            func (function): The function to decorate.
        Returns:
            function: The decorated function.
        """
        def wrapper(self, *args, **kwargs):
            if not self.fitted:
                raise InterpolatorError("Interpolator has not been fitted yet")
            return func(self, *args, **kwargs)
        return wrapper
    
    def fit(self, control_points:np.ndarray, weights:Optional[np.ndarray] = None) -> None:
        """Fits the Bezier curve to the given control points.
        Args:
            control_points (np.ndarray): The control points of the Bezier curve.
            weights (Optional[np.ndarray]): The weights of the control points. Defaults to None.
        """
        self.degree             = len(control_points) - 1
        self.bezier_curve       = BezierClass(control_points = control_points, degree = self.degree)
        self.control_points     = control_points
        self.normalize_factor   = np.max(control_points)

        #Compute the t value lookup table based on the number of arc length samples
        self.bezier_curve.compute_lookup_table(self.arc_length_samples)
        self.fitted             = True
    
    def is_fitted(self) -> bool:
        """Checks if the interpolator is fitted.
        Returns:
            bool: True if the interpolator is fitted, False otherwise.
        """
        return self.fitted
    
    @fitted_method
    def interpolate(self, points:np.ndarray, normalize:bool = False) -> np.ndarray:
        """Interpolates the given points using the generated Bezier curve.
        Args:
            points (np.ndarray): The percentage points to interpolate.
            normalize (bool): Whether to normalize the points to be between 0 and 1. Defaults to False.
        Returns:
            np.ndarray: The interpolated points.
        
        Raises:
            InterpolatorError: If the points provided are not within the range [0,1] and not normalized.
        """
        #Normalize the points if requested, and check that they are within the range [0,1] if not normalized
        if normalize:
            normalizing_factor  = np.max(self.control_points)
            points              = points / normalizing_factor
        else:
            #Check that the points are between 0 and 1
            if np.min(points) < 0 or np.max(points) > 1:
                raise InterpolatorError("Given points must be in the range [0,1]")
            else:
                normalizing_factor = 1
        
        #Find the corresponding t-values for the given points
        t_values = [self.bezier_curve.lookup_nearest_t_value(point) for point in points]

        #Interpolate the points using the t-values, and denormalize them if necessary
        return self.bezier_curve.sample_points(t_values) * normalizing_factor
    
    @fitted_method
    def sample(self, num_samples:int) -> np.ndarray:
        """Samples the Bezier curve at the given number of points, evenly spaced by arc length.
        Args:
            num_samples (int): The number of points to sample.
        Returns:
            np.ndarray: The sampled points.
        """
        return self.bezier_curve.sample_points_evenly_spaced(num_samples)
    
    @fitted_method
    def get_value(self, t:float) -> float:
        """Returns the value of the Bezier curve at the given t-value, by arc length interpolation.
        Args:
            t (float): The t-value to evaluate the Bezier curve at.
        Returns:
            float: The value of the Bezier curve at the given t-value.
        """
        t_value = self.bezier_curve.lookup_nearest_t_value(t)
        return self.bezier_curve.evaluate(t_value)
    
    @fitted_method
    def get_derivative(self, t:float, order:Optional[int] = 1) -> float:
        """Returns the derivative of the Bezier curve at the given t-value, by arc length interpolation.
        Args:
            t (float): The t-value to evaluate the Bezier curve at.
        Returns:
            float: The derivative of the Bezier curve at the given t-value.
        """
        derivative_curve = self.bezier_curve.derivative(order=order)
        t_value = derivative_curve.lookup_nearest_t_value(t)
        return derivative_curve.evaluate(t_value)

class LeastSquaresLinearInterpolator(InterpolatorMixin):
    """Generates a linear interpolation model from the given control points.
       PROS:
        - Easy to understand
        - Can be used for interpolation of any n-dimensional curve
        - When control points are close together, the curve will be close to the control points
        - Scalable with the number of points
       CONS:
        - Unreliable for curves with high curvature
        - Linear
        - Only a single derivative is available
    """

    def __init__(self) -> None:
        self.control_points = None
        self.weights        = None
        self.model          = None
        self.fitted         = False
    
    def fitted_method(func):
        """Decorator for methods that require the interpolator to be fitted.
        Args:
            func (function): The function to decorate.
        Returns:
            function: The decorated function.
        """
        def wrapper(self, *args, **kwargs):
            if not self.fitted:
                raise InterpolatorError("Interpolator has not been fitted yet")
            return func(self, *args, **kwargs)
        return wrapper
    
    def fit(self, control_points:np.ndarray, weights:Optional[np.ndarray] = None) -> None:
        """Fits the linear model to the given control points.
        Args:
            control_points (np.ndarray): The control points of the linear model.
            weights (Optional[np.ndarray]): The weights of the control points. Defaults to None.
        """
        self.control_points = control_points
        self.weights        = weights
        self.model          = LinearRegression()
        self.model.fit(self.weights, self.control_points)
        self.fitted         = True
    
    def is_fitted(self) -> bool:
        """Checks if the interpolator is fitted.
        Returns:
            bool: True if the interpolator is fitted, False otherwise.
        """
        return self.fitted
    
    @fitted_method
    def interpolate(self, points:np.ndarray, normalize:bool = False) -> np.ndarray:
        """Interpolates the given points using the generated linear model.
        Args:
            points (np.ndarray): The percentage points to interpolate.
            normalize (bool): Unused parameter, kept for compatibility with the InterpolatorMixin. Defaults to False.
        Returns:
            np.ndarray: The interpolated points.
        
        Raises:
            InterpolatorError: If the points provided are not within the range [0,1] and not normalized.
        """
        #Reshape the points to be a column vector
        points = np.array(points).reshape(-1,1)
        return self.model.predict(points)
    
    @fitted_method
    def sample(self, num_samples:int) -> np.ndarray:
        """Samples the linear model at the given number of points, evenly spaced from the smallest to the largest.
        Args:
            num_samples (int): The number of points to sample.
        Returns:
            np.ndarray: The sampled points.
        """
        predict_points = np.linspace(np.min(self.weights), np.max(self.weights), num_samples).reshape(-1,1)
        return self.model.predict(predict_points)
    
    @fitted_method
    def get_value(self, t:float) -> float:
        """Returns the value of the linear model at the given t-value, by linear interpolation.
        Args:
            t (float): The t-value to evaluate the linear model at.
        Returns:
            float: The value of the linear model at the given t-value.
        """
        return self.model.predict([t])[0]
    
    @fitted_method
    def get_derivative(self, t:float, order:Optional[int] = 1) -> float:
        """Returns the derivative of the linear model at the given t-value, by linear interpolation.
        Args:
            t (float): The t-value to evaluate the linear model at.
        Returns:
            float: The derivative of the linear model at the given t-value.
        """
        return self.model.coef_[order]