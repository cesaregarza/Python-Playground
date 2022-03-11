from typing import Optional, Union
import numpy as np
import numpy.typing as npt
import pandas as pd
from enum import Enum, auto
from warnings import warn

class Directional_Format(Enum):
    cartesian   = auto()
    polar       = auto()

class DirectionalStatistics:

    def __init__(self, data:    npt.ArrayLike,
                       format:  Union[Directional_Format, str]  = Directional_Format.cartesian,
                       center:  tuple[float, float]             = (0, 0),
                       base:    Union[float, int]               = 2 * np.pi) -> None:
        """Object that handles directional statistics for a given array of data.

        Args:
            data (npt.ArrayLike): Array-like object containing data.
            format(Union[Directional_Format, str]): Format of the data.
            center(tuple[float, float]): Center to use for offsetting the data. If cartesian, (x, y) is used. If polar, (r, theta) is used. Defaults to (0, 0) for both.
            base(Union[float, int]): Directional base to use for converting the data. Defaults to 2 * pi.
        """

        #Validate the data
        data = self.__validate_data(data)

        #Validate the format
        format = self.__validate_format(format)
        
        #Return an error if the number of columns is not exactly 2
        if data.shape[1] != 2:
            raise ValueError("Data must have exactly 2 columns.")

        #Set the center of the data if it's not provided according to the format
        self.center             = (0,0) if center is None else center
        self.raw_data           = data
        self.format             = format
        self.base               = base
        self.base_to_radians    = base / (2 * np.pi)
        self.normalize_format()
    
    @property
    def x(self) -> np.ndarray:
        """Array of x-coorinates of the data

        Returns:
            np.ndarray: Array of x-coorinates of the data
        """
        return self.cartesian_data[:, 0]
    
    @property
    def y(self) -> np.ndarray:
        """Array of y-coorinates of the data

        Returns:
            np.ndarray: Array of y-coorinates of the data
        """
        return self.cartesian_data[:, 1]
    
    @property
    def r(self) -> np.ndarray:
        """Array of radial distances of the data

        Returns:
            np.ndarray: Array of radial distances of the data
        """
        return self.polar_data[:, 0]
    
    @property
    def theta(self) -> np.ndarray:
        """Array of angular components of the data in the directional base e.g. 360 degrees.

        Returns:
            np.ndarray: Array of angular components of the data
        """
        return self.polar_data[:, 1] * self.base_to_radians
    
    @property
    def radians(self) -> np.ndarray:
        """Array of angular components of the data in radians

        Returns:
            np.ndarray: Array of angular components of the data in radians
        """
        return self.polar_data[:, 1]
    
    @property
    def mean_cartesian(self) -> tuple[float, float]:
        """Circular mean of the data.

        Returns:
            tuple[float, float]: Mean of the data.
        """
        return DirectionalStatistics.cartesian_mean(data = self.cartesian_data, input_format = Directional_Format.cartesian)
    
    @property
    def mean_theta(self) -> float:
        """Mean of the angular components of the data.

        Returns:
            float: Mean of the angular components of the data.
        """
        x, y = self.mean_cartesian
        return np.arctan2(y, x) * self.base_to_radians
    
    @property
    def mean_radians(self) -> float:
        """Mean of the angular components of the data in radians.

        Returns:
            float: Mean of the angular components of the data in radians.
        """
        return self.mean_theta / self.base_to_radians

    @property
    def var(self) -> float:
        """Variance of the data.

        Returns:
            float: Variance of the data.
        """
        x, y = self.mean_cartesian
        r = np.sqrt(x**2 + y**2)
        r = min(1, r)
        return 1 - r

    def __validate_data(self, data:npt.ArrayLike) -> npt.ArrayLike:
        """Validates the data.

        Args:
            data (npt.ArrayLike): Array-like object containing data.

        Returns:
            npt.ArrayLike: Array-like object containing data.
        """

        #If the data is a list, make sure it is a rectangular array where all rows are the same length
        if isinstance(data, list):
            #Make sure the length of each element in the list is the same
            if not all(len(x) == len(data[0]) for x in data):
                raise ValueError("All sublists must be the same length.")
            
            #Convert the list to a numpy array
            data = np.array(data)
        
        #Make sure the elements of the data are numeric
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("Data must be numeric.")
        
        #Return a warning if there are duplicate values
        if len(data) != len(np.unique(data)):
            warn("Duplicate values detected. This may cause errors in the calculations.")
        
        #Raise an error if there are any NaN values
        if np.isnan(data).any():
            raise ValueError("Data cannot contain NaN values.")
        
        #Return the data
        return data
    
    def __validate_format(self, format:Union[Directional_Format, str]) -> Directional_Format:
        """Validates the format.

        Args:
            format (Union[Directional_Format, str]): Format of the data.

        Returns:
            Directional_Format: Format of the data.
        """

        #If the format is a string, turn it into a Directional_Format
        if isinstance(format, str):
            try:
                format = Directional_Format[format]
            except KeyError:
                raise ValueError("Format must be a valid Directional_Format.")
        
        #Return the format
        return format
    
    def normalize_format(self) -> None:
        """Normalizes the data according to the format.
        """

        if self.format == Directional_Format.cartesian:
            self.cartesian_data     = self.raw_data - self.center
            r                       = np.sqrt(self.x**2 + self.y**2)
            theta                   = np.arctan2(self.y, self.x)
            self.polar_data         = np.array([r, theta]).T

        elif self.format == Directional_Format.polar:
            self.polar_data         = self.raw_data
            x                       = self.r * np.cos(self.theta)
            y                       = self.r * np.sin(self.theta)
            self.cartesian_data     = np.array([x, y]).T
    
        return
    
    def change_base(self, new_base:float) -> None:
        """Changes the base of the data.

        Args:
            new_base (float): New base of the data.
        """

        self.base               = new_base
        self.base_to_radians    = new_base / (2 * np.pi)
    
    @staticmethod
    def convert(data: npt.ArrayLike, input_format:Union[Directional_Format, str], output_format:Union[Directional_Format, str], base:Union[float, int] = 2 * np.pi) -> npt.ArrayLike:
        """Converts the data from one format to another.

        Args:
            data (npt.ArrayLike): Array-like object containing data.
            input_format(Union[Directional_Format, str]): Format of the data.
            output_format(Union[Directional_Format, str]): Format of the data.
            base(Union[float, int]): Directional base to use for converting the data. Defaults to 2 * pi.

        Returns:
            npt.ArrayLike: Array-like object containing data.
        """
        if isinstance(output_format, str):
            output_format = Directional_Format[output_format]
        
        if isinstance(input_format, str):
            input_format = Directional_Format[input_format]

        #If the input and output formats are the same, return the data
        if input_format == output_format:
            return data
        
        #If the output format is cartesian, then the input format must be polar
        if output_format == Directional_Format.cartesian:
            return DirectionalStatistics.cartesian_to_polar(data, base)
        elif output_format == Directional_Format.polar:
            return DirectionalStatistics.polar_to_cartesian(data, base)
    
    @staticmethod
    def cartesian_to_polar(cartesian_data:npt.ArrayLike, base:Union[float, int] = 2 * np.pi) -> np.ndarray:
        """Converts cartesian data to polar data.

        Args:
            cartesian_data (npt.ArrayLike): Array-like object containing cartesian data.
            base (Union[float, int]): Directional base to use for converting the data. Defaults to 2 * pi.

        Returns:
            np.ndarray: Array-like object containing polar data.
        """

        base_to_radians = base / (2 * np.pi)
        x               = cartesian_data[:, 0]
        y               = cartesian_data[:, 1]
        r               = np.sqrt(x**2 + y**2)
        theta           = np.arctan2(y, x) * base_to_radians

        return np.array([r, theta]).T
    
    @staticmethod
    def polar_to_cartesian(polar_data:npt.ArrayLike, base:Union[float, int] = 2 * np.pi) -> np.ndarray:
        """Converts polar data to cartesian data.

        Args:
            polar_data (npt.ArrayLike): Array-like object containing polar data.
            base (Union[float, int]): Directional base to use for converting the data. Defaults to 2 * pi.

        Returns:
            np.ndarray: Array-like object containing cartesian data.
        """

        base_to_radians = base / (2 * np.pi)
        r               = polar_data[:, 0]
        theta           = polar_data[:, 1] / base_to_radians
        x               = r * np.cos(theta)
        y               = r * np.sin(theta)

        return np.array([x, y]).T
    
    @staticmethod
    def project_to_unit_circle(x:npt.ArrayLike, y:npt.ArrayLike) -> np.ndarray:
        """Projects cartesian data to the unit circle.

        Args:
            x (npt.ArrayLike): Array-like object containing x-coordinates of the data.
            y (npt.ArrayLike): Array-like object containing y-coordinates of the data.

        Returns:
            np.ndarray: Array-like object containing cartesian data.
        """

        r = np.sqrt(x**2 + y**2)
        x = x / r
        y = y / r

        return np.array([x, y]).T
    
    @staticmethod
    def distance_between_angles(theta1:npt.ArrayLike, theta2:npt.ArrayLike, base:Union[float, int] = 2 * np.pi) -> np.ndarray:
        """Calculates the distance between two angles, compensating for wrap-around.

        Args:
            theta1 (npt.ArrayLike): Array-like object containing the first angle.
            theta2 (npt.ArrayLike): Array-like object containing the second angle.
            base (Union[float, int]): Directional base to use. Defaults to 2 * pi.

        Returns:
            np.ndarray: Array-like object containing the distance between the angles in the original base.
        """

        theta1          = np.array(theta1)
        theta2          = np.array(theta2)
        naive_distance  = np.abs(theta1 - theta2).reshape((-1, 1))
        where           = np.where(naive_distance > base / 2)

        if len(where[0]) > 0:
            for row, col in where:
                naive_distance[row, col] = base - naive_distance[row, col]
        
        return naive_distance
    
    @staticmethod
    def cartesian_mean(data: npt.ArrayLike, input_format:Union[Directional_Format, str], base:Union[float, int] = 2 * np.pi) -> tuple[float, float]:
        """Calculates the mean of the data, converting it to cartesian coordinates if necessary.

        Args:
            data (npt.ArrayLike): Array-like object containing data.
            input_format(Union[Directional_Format, str]): Format of the data.
            base(Union[float, int]): Directional base to use for converting the data. Defaults to 2 * pi.

        Returns:
            npt.ArrayLike: Array-like object containing the mean of the components of the data.
        """
        data = np.array(data)

        #Convert the data to cartesian coordinates
        cartesian_data = DirectionalStatistics.convert(data, input_format, Directional_Format.cartesian, base)
        x              = cartesian_data[:, 0]
        y              = cartesian_data[:, 1]
        projected      = DirectionalStatistics.project_to_unit_circle(x, y)
        x              = projected[:, 0]
        y              = projected[:, 1]

        return np.mean(x), np.mean(y)
    
    @staticmethod
    def from_dataframe(dataframe:pd.DataFrame, 
                       columns:Optional[list] = None,
                       format:Union[Directional_Format, str, None] = None,
                       x: Optional[str] = None,
                       y: Optional[str] = None,
                       r: Optional[str] = None,
                       theta: Optional[str] = None,
                       center: Union[None, float, tuple] = None,
                       base:float = 2 * np.pi) -> 'DirectionalStatistics':
        """Creates a DirectionalStatistics object from a pandas DataFrame.

        Args:
            dataframe (pd.DataFrame): DataFrame containing data.
            columns (Optional[list]): List of columns to use. Defaults to None.
            format (Union[Directional_Format, str, None]): Format of the data. Defaults to None.
            x (Optional[str]): Column containing x-coordinates. Defaults to None.
            y (Optional[str]): Column containing y-coordinates. Defaults to None.
            r (Optional[str]): Column containing radial coordinates. Defaults to None.
            theta (Optional[str]): Column containing angular coordinates. Defaults to None.
            center (Union[None, float, tuple]): Center of the data. Defaults to None.
            base (float): Directional base to use for converting the data. Defaults to 2 * pi.

        Returns:
            DirectionalStatistics: DirectionalStatistics object containing the data.
        """
        
        #If columns are specified, make sure the format is specified
        if columns is not None:
            if format is None:
                raise ValueError("If columns are specified, the format must be specified.")
            
            col_1, col_2 = columns
        #Else, check if either both x and y or r and theta are specified
        else:
            xy = (x is not None) and (y is not None)
            rt = (r is not None) and (theta is not None)
            if not (xy or rt):
                if len(dataframe.columns) == 2:
                    col_1, col_2 = dataframe.columns
                    format = format if format is not None else Directional_Format.cartesian
                else:
                    raise ValueError("Either both x and y or r and theta must be specified if dataframe has more than 2 columns.")
            elif xy:
                col_1   = x
                col_2   = y
                format  = Directional_Format.cartesian
            elif rt:
                col_1   = r
                col_2   = theta
                format  = Directional_Format.polar
        
        data = dataframe[[col_1, col_2]].values
        return DirectionalStatistics(data, format, center, base)