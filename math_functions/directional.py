from typing import Union
import numpy as np
import numpy.typing as npt
from enum import Enum, auto

class Direction_Format(Enum):
    cartesian   = auto()
    polar       = auto()

class DirectionalStatistics:

    def __init__(self, data:npt.ArrayLike, 
                       format:Direction_Format = Direction_Format.cartesian,
                       center:Union[tuple,float,None] = None,
                       base:Union[float, int] = 2 * np.pi) -> None:
        """Object that handles directional statistics for a given array of data. Must be 2D.

        Args:
            data (npt.ArrayLike): Numpy or Array-like object containing data.
            format (Direction_Format, optional): Format of the data. Defaults to cartesian.
            center (Optional[tuple], optional): Center of the data. Defaults to None.
            base (Union[float, int], optional): Angular base for the directional statistics. Defaults to 2 * pi.
        """
        #Check that the data is 2D
        if data.ndim != 2:
            raise NotImplementedError("Directional statistics only supports 2D data at this time.")

        #Set the center of the data if it is not given according to the format
        if center is None:
            if format == Direction_Format.cartesian:
                center = (0, 0)
            elif format == Direction_Format.polar:
                center = 0
        
        #Translate the data according to the center provided if the format is cartesian
        if format == Direction_Format.cartesian:
            data = data - center
        
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
            self.radians    = np.arctan2(self.y, self.x)

        elif self.format    == Direction_Format.polar:
            self.radians    = self.raw_data / self.base_to_radians
            self.cartesian  = [np.cos(self.radians), np.sin(self.radians)]

        return
    
    @staticmethod
    def convert(data: npt.ArrayLike, input:str = "polar", output:str = "cartesian", base:Union[float, int] = 2 * np.pi) -> np.ndarray:
        """Converts the data from one format to another.

        Args:
            data (npt.ArrayLike): Numpy or Array-like object containing data.
            input (str, optional): Input format of the data. Defaults to polar.
            output (str, optional): Output format of the data. Defaults to cartesian.
            base (Union[float, int], optional): Angular base for the directional statistics. Defaults to 2 * pi.
        
        Returns:
            np.ndarray: Data in the specified format.
        """
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
            np.ndarray: Array of cartesian coordinates in the form (x, y).
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
            np.ndarray: Array of polar coordinates in the form (r, theta).
        """
        base_to_radians = base / (2 * np.pi)
        rad             = np.arctan2(cartesian[:, 1], cartesian[:, 0])
        angles          = (rad * base_to_radians) % base

        #Calculate the vector length
        lengths         = np.linalg.norm(cartesian, axis=1)
        return np.array([lengths, angles])
    
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
            theta2 (npt.ArrayLike): Array of second angles.
            base (float, optional): Base for the directional statistics. Defaults to 2 * pi.
        
        Returns:
            np.ndarray: Array of distances between the angles.
        """
        theta1          = np.array(theta1)
        theta2          = np.array(theta2)
        naive_distance  = np.abs(theta1 - theta2).reshape((-1, 1))
        where           = np.where(naive_distance > base / 2)

        if len(where[0]) > 0:
            where_row, where_col = where
            for row, col in zip(where_row, where_col):
                naive_distance[row, col] = base - naive_distance[row, col]
        
        return naive_distance

    def mean(self, output:str = "intrinsic") -> np.ndarray:
        """Calculates the mean of the data, by taking the mean of the respective cartesian coordinates

        Args:
            output(str): Output format of the extrinsic mean. Valid options are cartesian, radians, polar, extrinsic, and intrinsic.
        
        Returns:
            np.ndarray: Mean of the data in the specified format.
        """
        return DirectionalStatistics.circular_mean(self.cartesian, base=self.base, output=output)
    
    @staticmethod
    def circular_mean(data: npt.ArrayLike, base:float = 2 * np.pi, input:str = "cartesian", output:str = "intrinsic") -> np.ndarray:
        """Calculates the mean of the data, by taking the mean of the respective cartesian components of the data.

        Args:
            data (npt.ArrayLike): Array of data, must be 2D.
            base (float, optional): Base for the directional statistics. Defaults to 2 * pi.
            input (str, optional): Input format of the data. Valid options are cartesian, radians, or polar. Defaults to "cartesian".
            output (str, optional): Output format of the data. Valid options are cartesian, radians, polar, extrinsic, or intrinsic. Defaults to "intrinsic".

        Returns:
            np.ndarray: [description]
        """
        convert_dict = {
            "extrinsic": "cartesian",
            "intrinsic": "polar"
        }
        if output in convert_dict:
            output = convert_dict[output]
        
        #Convert the data to cartesian coordinates
        data = DirectionalStatistics.convert(np.array(data), input, "cartesian", base)
        x_mean, y_mean = DirectionalStatistics.__component_mean(data)

        #Convert the calculated extrinsic mean to the desired output format using the convert method
        kwargs = {
            "data": np.array([x_mean, y_mean]).reshape((1, 2)),
            "input": "cartesian",
            "output": output,
            "base": base
        }
        return DirectionalStatistics.convert(**kwargs).flatten()
    
    @staticmethod
    def __component_mean(data: npt.ArrayLike) -> tuple[float, float]:
        """Calculates the mean of the data, by taking the mean of the respective cartesian components of the data.

        Args:
            data (npt.ArrayLike): Array of data, must be 2D and in cartesian coordinates.

        Returns:
            tuple[float, float]: (x_mean, y_mean)
        """
        data = np.array(data)
        x, y = data[:, 0], data[:, 1]
        x, y = DirectionalStatistics.project_to_unit_circle(x, y)
        return np.mean(x), np.mean(y)
    
    @staticmethod
    def circmean(data: npt.ArrayLike) -> float:
        """Calculates the circular mean of the data.

        Args:
            data (npt.ArrayLike): Array of data, must be 2D.

        Returns:
            float: Intrinsic mean of the data, in radians.
        """

        x_mean, y_mean = DirectionalStatistics.__component_mean(data)
        return np.arctan2(y_mean, x_mean)
    
    @staticmethod
    def circvar(data: npt.ArrayLike) -> float:
        """Calculates the circular variance of the data.

        Args:
            data (npt.ArrayLike): Array of data, must be 2D.

        Returns:
            float: Circular variance of the data.
        """
        x_mean, y_mean = DirectionalStatistics.__component_mean(data)
        return 1.0 - np.mean(np.sqrt(x_mean**2 + y_mean**2))
    
    @property
    def x(self) -> np.ndarray:
        """Array of x-coordinates of the data.

        Returns:
            np.ndarray: Array of x-coordinates of the data.
        """
        return self.cartesian[:, 0]
    
    @property
    def y(self) -> np.ndarray:
        """Array of y-coordinates of the data.

        Returns:
            np.ndarray: Array of y-coordinates of the data.
        """
        return self.cartesian[:, 1]
    
    @property
    def r(self) -> np.ndarray:
        """Array of radial coordinates of the data.

        Returns:
            np.ndarray: Array of radial coordinates of the data.
        """
        return np.sqrt(self.x**2 + self.y**2)
    
    @property
    def theta(self) -> np.ndarray:
        """Array of angular coordinates of the data, in the angular base.

        Returns:
            np.ndarray: Array of angular coordinates of the data.
        """
        return self.radians * self.base_to_radians
    
    @property
    def polar(self) -> np.ndarray:
        """Array of polar coordinates of the data, in the angular base.

        Returns:
            np.ndarray: Array of polar coordinates of the data.
        """
        return np.array([self.r, self.theta]).T
    
    @property
    def polar_radians(self) -> np.ndarray:
        """Array of polar coordinates of the data, in radians.

        Returns:
            np.ndarray: Array of polar coordinates of the data.
        """
        return np.array([self.r, self.radians]).T
    
    @property
    def var(self) -> np.ndarray:
        """Array of variance of the data.

        Returns:
            np.ndarray: Array of variance of the data.
        """
        return self.circvar(self.cartesian)
    
    def change_base(self, base:float) -> None:
        """Changes the base of the data.

        Args:
            base (float): New base for the data.
        """
        self.base               = base
        self.base_to_radians    = base / (2 * np.pi)