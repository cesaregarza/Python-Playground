from typing import Union, Optional
from warnings import warn

from .enums import ColorFormats, ColorObjects
from colormath.color_conversions import convert_color as colormath_convert_color
from colormath.color_objects import ColorBase
from webcolors import name_to_hex
import numpy as np
from daltonize.daltonize import simulate

def format_to_object(color_format: ColorFormats) -> ColorBase:
    return ColorObjects[color_format.value].value

def convert_color(input_color: Union[np.ndarray, ColorBase, list[int], tuple[int, int, int]],
                  output_format: ColorFormats,
                  input_format: Optional[ColorFormats] = None) -> np.ndarray:
    """Converts a single color from one format to another.
    Args:
        input_color (np.ndarray): The color to convert.
        input_format (ColorFormats): The format of the input color.
        output_format (ColorFormats): The format to convert the color to.
    Returns:
        np.ndarray: The converted color as a numpy array.
    """

    #If the input format is not specified, make sure the input color is a ColorBase instance
    if input_format is None:
        if isinstance(input_color, ColorBase):
            input_format = input_color.__class__
        else:
            raise ValueError("Input format not specified and input color is not a ColorBase instance")
    
    #If the input and output formats are the same, return the input color
    if input_format == output_format:
        return input_color
    
    if isinstance(output_format, ColorFormats):
        output_format = format_to_object(output_format)

    #If the input color is a numpy array, flatten it
    if isinstance(input_color, np.ndarray):
        input_color = input_color.flatten()
    
    #Drop the alpha channel if it exists, if input_color is not a ColorBase instance
    

    #If the input color is a numpy array, convert it to a colormath color object
    if isinstance(input_color, (np.ndarray, list, tuple)):
        if len(input_color) == 4:
            color = input_color[:3]
        elif len(input_color) == 3:
            color = input_color
        else:
            raise ValueError("Invalid color: " + str(input_color))

        #Create the appropriate color format object
        color = format_to_object(input_format)(*color, is_upscaled=True)
    #Otherwise, check if the input color is a colormath color object
    elif not isinstance(input_color, ColorBase):
        raise TypeError("Invalid input color")
    else:
        color = input_color

    #Convert the color to the output format
    output_color = colormath_convert_color(color, output_format, is_upscaled=True)

    return output_color.get_value_tuple()

def rgb_hex_to_tuple(color:str) -> tuple[int, int, int]:
    """Converts a hexadecimal color to a list of integers.
    Args:
        color (str): The hexadecimal color to convert.
    Returns:
        list[int]: The converted color as a list of integers.
    """

    if len(color) == 4:
        color = color[0] + color[1] + color[1] + color[2] + color[2] + color[3] + color[3]
    elif len(color) == 6 and color[0] != "#":
        color = "#" + color
    elif len(color) != 7:
        raise ValueError("Invalid color string")
    
    #Convert the color to an integer from base 16
    parsed_color = [int(color[i:i+2], 16) for i in range(1, len(color),2)]

    return tuple(parsed_color)

def rgb_tuple_to_hex(color: list[int], upscaled:bool = True) -> str:
    """Converts a list of integers to a hexadecimal color.
    Args:
        color (list[int]): The color to convert.
        upscaled (bool): Whether or not the input color is upscaled.
    Returns:
        str: The converted color as a hexadecimal string.
    """

    if not upscaled:
        color = [int(c * 255) for c in color]
    
    #Give warnings for likely unwanted behavior
    if all(c <= 1 for c in color) and upscaled:
        warn("Likely unwanted behavior: color values are all below 1 and upscaled is set to True")
    if all(c >= 254 for c in color) and not upscaled:
        warn("Likely unwanted behavior: color values are all above 254 and upscaled is set to False")
    
    #Make sure the values are within the range of 0-255
    color = [int(c) if c <= 255 else 255 for c in color]
    color = [int(c) if c >= 0   else 0   for c in color]

    return "#" + "".join(["{:02x}".format(c) for c in color]).upper()

def rgb_to_luminosity(color: list[int]) -> float:
    """Converts a list of RGB values to a luminosity value.
    Args:
        color (list[int]): The color to convert.
    Returns:
        float: The luminosity value.
    """

    return (0.2126 * color[0]) + (0.7152 * color[1]) + (0.0722 * color[2])

def rgb_to_luminosity_hex(color: str) -> str:
    """Converts a hexadecimal color to a luminosity hexadecimal.
    Args:
        color (str): The hexadecimal color to convert.
    Returns:
        str: The luminosity hexadecimal value.
    """

    #Convert the color to a list of integers and convert it to a luminosity value
    luminosity = rgb_to_luminosity(rgb_hex_to_tuple(color))

    #Duplicate the luminosity value for each channel, then convert it to a hexadecimal string
    return rgb_tuple_to_hex([luminosity, luminosity, luminosity])

def parse_rgb(color: Union[str, list[int], tuple[int, int, int]]) -> tuple[int, int, int]:
    """Parse colors from a variety of formats into a list of RGB integers
    Args:
        color (Union[str, list[int]]): The color to parse.
    Returns:
        list[int]: list containing values for RGB channels
    """

    #If the color is a string, check if it is a hexadecimal or a color name
    if isinstance(color, str):
        if color[0] == "#":
            return rgb_hex_to_tuple(color)
        else:
            try:
                color_hex = name_to_hex(color).upper()
                return rgb_hex_to_tuple(color_hex)
            except ValueError:
                raise ValueError("Invalid color string: " + color)
    #If the color is a list or tuple, check if it is a list of RGB values
    elif isinstance(color, (list, tuple)):
        if len(color) == 3:
            return tuple(color)
        elif len(color) == 4:
            return tuple(color[:3])
        else:
            raise ValueError("Invalid color: " + str(color))
    else:
        raise ValueError("Invalid color: " + str(color))

def daltonize(rgb_tuples:Union[list[tuple[int, int, int]], tuple[tuple[int, int, int]], np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Daltonize a list of RGB tuples
    Args:
        rgb_tuples (list[tuple[int, int, int]]): List of RGB tuples to daltonize
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Deut, prot, and trit daltonized RGB tuples
    """
    length = len(rgb_tuples)

    rgb_array = np.array(rgb_tuples).reshape((length, 1, 3))

    #Daltonize the array
    deut_tuples = np.rint(simulate(rgb_array, "d")).reshape((length, 3)).astype(int)
    prot_tuples = np.rint(simulate(rgb_array, "p")).reshape((length, 3)).astype(int)
    trit_tuples = np.rint(simulate(rgb_array, "t")).reshape((length, 3)).astype(int)

    return deut_tuples, prot_tuples, trit_tuples