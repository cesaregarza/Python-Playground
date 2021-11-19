from enum import Enum
from colormath import color_objects

class ColorFormats(Enum):
    RGB     = "RGB"
    RGBA    = "RGBA"
    HSL     = "HSL"
    HSLA    = "HSLA"
    HSV     = "HSV"
    HSVA    = "HSVA"
    LAB     = "LAB"

class ColorObjects(Enum):
    RGB    = color_objects.sRGBColor
    RGBA   = color_objects.sRGBColor
    HSL    = color_objects.HSLColor
    HSLA   = color_objects.HSLColor
    HSV    = color_objects.HSVColor
    HSVA   = color_objects.HSVColor
    LAB    = color_objects.LabColor