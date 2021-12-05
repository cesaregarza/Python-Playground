from typing import Union
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from enum import Enum, auto

class Direction_Format(Enum):
    cartesian   = auto()
    polar       = auto()

class DirectionalStatistics:

    def __init__(self, data:npt.ArrayLike, 
                       format:Direction_Format = Direction_Format.cartesian,
                       base:Union[float, int] = 2 * np.pi) -> None:
        """Object that handles directional statistics for a given array of data. Must be 2D.

        Args:
            data (npt.ArrayLike): Numpy or Array-like object containing data.
            base (Union[float, int], optional): Base for the directional statistics. Defaults to 2 * pi.
        """
        self.raw_data           = np.array(data)
        self.format             = format
        self.base               = base
        self.base_to_radians    = base / (2 * np.pi)
        self.normalize_format()
    
    def normalize_format(self) -> None:
        """Normalizes the data from the given format to radians.
        """

        if self.format    == Direction_Format.cartesian:
            self.cartesian  = self.raw_data
            self.radians    = np.arctan2(self.cartesian[:, 1], self.cartesian[:, 0])

        elif self.format    == Direction_Format.polar:
            self.radians    = self.raw_data / self.base_to_radians
            self.cartesian  = [np.cos(self.radians), np.sin(self.radians)]

        return
    
    @staticmethod
    def convert(data: npt.ArrayLike, input:str = "polar", output:str = "cartesian", base:Union[float, int] = 2 * np.pi) -> np.ndarray:
        if input == output:
            return data
        elif output == "cartesian":
            return DirectionalStatistics.convert_to_cartesian(data, base)
        elif output == "polar":
            return DirectionalStatistics.convert_to_polar(data, base)
    
    @staticmethod
    def convert_to_cartesian(radians:npt.ArrayLike, base:Union[float, int] = 2 * np.pi) -> np.ndarray:
        """Converts a given array of radians to cartesian coordinates.

        Args:
            radians (npt.ArrayLike): Array of radians.
            base (Union[float, int], optional): Base for the directional statistics. Defaults to 2 * pi.
        
        Returns:
            np.ndarray: Array of cartesian coordinates.
        """
        base_to_radians     = base / (2 * np.pi)
        converted_radians   = radians / base_to_radians
        x, y                = np.cos(converted_radians), np.sin(converted_radians)
        return np.array([x, y])

    @staticmethod
    def convert_to_polar(cartesian:npt.ArrayLike, base:Union[float, int] = 2 * np.pi) -> np.ndarray:
        """Converts a given array of cartesian coordinates to polar coordinates.

        Args:
            cartesian (npt.ArrayLike): Array of cartesian coordinates.
            base (Union[float, int], optional): Base for the directional statistics. Defaults to 2 * pi.
        
        Returns:
            np.ndarray: Array of polar coordinates.
        """
        base_to_radians = base / (2 * np.pi)
        rad             = np.arctan2(cartesian[:, 1], cartesian[:, 0])
        return rad * base_to_radians
    
    @staticmethod
    def project_to_unit_circle(x:npt.ArrayLike, y:npt.ArrayLike) -> np.ndarray:
        """Projects a given array of cartesian coordinates to the unit circle.

        Args:
            x (npt.ArrayLike): Array of x-coordinates.
            y (npt.ArrayLike): Array of y-coordinates.
        
        Returns:
            np.ndarray: Array of cartesian coordinates projected to the unit circle.
        """
        r = np.sqrt(x**2 + y**2)
        return x / r, y / r
    
    @staticmethod
    def distance_between_angles(theta1:npt.ArrayLike, theta2:npt.ArrayLike, base:float = 2 * np.pi) -> np.ndarray:
        """Calculates the distance between two angles, compensating for wrap-around.

        Args:
            theta1 (npt.ArrayLike): Array of first angles.
            theta2 (float): Array of second angles.
            base (float, optional): Base for the directional statistics. Defaults to 2 * pi.
        
        Returns:
            np.ndarray: Array of distances between the angles.
        """
        naive_distance = np.abs(theta1 - theta2)
        where = np.where(naive_distance > base / 2)
        if len(where[0]) > 0:
            where_row, where_col = where
            for row, col in zip(where_row, where_col):
                naive_distance[row, col] = base - naive_distance[row, col]
        
        return naive_distance


    def extrinsic_mean(self, output:str = "cartesian") -> np.ndarray:
        """Calculates the extrinsic mean of the data, by taking the mean of the respective cartesian coordinates

        Args:
            output(str): Output format of the extrinsic mean. Valid options are cartesian, radians, and polar.
        
        Returns:
            np.ndarray: Extrinsic mean of the data, in cartesian coordinates.
        """
        # Calculate the mean of the cartesian coordinates, this is the extrinsic mean
        x, y            = self.cartesian[:, 0], self.cartesian[:, 1]
        x, y            = DirectionalStatistics.project_to_unit_circle(x, y)
        x_mean, y_mean  = np.mean(x), np.mean(y)

        # Convert the calculated extrinsic mean to the desired output format using the convert method
        base = 2 * np.pi if output == "radians" else self.base
        kwargs = {
            "data": np.array([x_mean, y_mean]).reshape((1, 2)),
            "input": "cartesian",
            "output": output,
            "base": base
        }
        return DirectionalStatistics.convert(**kwargs)
    
    def instrinsic_mean(self, output:str = "polar") -> np.ndarray:
        """Calculates the intrinsic mean of the data
        
        Args:
            output(str): Desired output format
        
        Returns:

        """
        #Search for the minimum of the sum of the squared differences using the scipy.optimize.minimize method
        def sum_of_squared_differences(theta:np.ndarray) -> float:
            dist_array  = DirectionalStatistics.distance_between_angles(theta, self.radians, self.base)
            return np.sum(dist_array**2)
        
        #Generate the initial guess for the minimization. We use the extrinsic mean of the data and project it to the unit circle
        initial_theta = self.extrinsic_mean(output = "polar")[0]
        theta_opt = minimize(sum_of_squared_differences, initial_theta, bounds=[(0, self.base)]).x

        #Convert the optimal theta to the desired output format
        base = 2 * np.pi if output == "radians" else self.base
        kwargs = {
            "data": theta_opt,
            "input": "polar",
            "output": output,
            "base": base
        }
        return DirectionalStatistics.convert(**kwargs)