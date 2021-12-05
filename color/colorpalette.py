from typing import Callable, Optional, Union, Type
import numpy as np
from math_functions import nDegreeBezierInterpolator, InterpolatorMixin
from .color_functions import *
from PIL import Image, ImageDraw
from plotly import graph_objects as go

from enum import Enum
from .enums import ColorFormats

class HueDirection(Enum):
    SHORTEST            = "shortest"
    LONGEST             = "longest"
    CLOCKWISE           = "clockwise"
    COUNTERCLOCKWISE    = "counterclockwise"
    SHORT               = "shortest"
    LONG                = "longest"
    CW                  = "clockwise"
    CCW                 = "counterclockwise"
    NA                  = "none"
    NOTAPPLICABLE       = "none"

class ColorPalette:
    def __init__(self, color_reference_list: list[Union[str, tuple[str, float]]],
                       color_format: ColorFormats = ColorFormats.RGB,
                       weights: Optional[Union[list[float], np.ndarray]] = None,
                       interpolator: Type[InterpolatorMixin] = nDegreeBezierInterpolator) -> None:
        

        #If weights are not provided, generate linear weights
        self.linear = weights == None
        linear_weights = np.linspace(0, 1, len(color_reference_list), endpoint = True, dtype = float)
        
        #If weights are not provided, use linear weights. If they are, check if they're linear
        if self.linear:
            weights = linear_weights
        else:
            self.linear = np.allclose(weights, linear_weights)
        
        self.weights = np.array(weights).reshape(-1, 1)
        self.color_format = color_format

        self.interpolator = interpolator()

        #Parse the color reference list depending on the format
        self.color_reference_rgb = self.parse_colors(color_reference_list, color_format)

    def parse_colors(self, color_reference_list: list[Union[str, tuple[float, float, float], tuple[float, float, float, float]]],
                           color_format: ColorFormats = ColorFormats.RGB) -> list[str]:
        """Parse the color reference list into a list of RGB hex codes
        Args:
            color_reference_list (list[Union[str, tuple[float, float, float], tuple[float, float, float, float]]]): List of colors to be parsed
            color_format (ColorFormats, optional): Format of the colors. Defaults to ColorFormats.RGB.
            weights (Optional[list[float]], optional): Weights to be used to interpolate the colors. Defaults to None.
        Returns:
            list[str]: List of RGB hex codes
        """
        
        #Parse the color reference list if it's RGB Hex, otherwise return the input color
        if (color_format == ColorFormats.RGB or color_format == ColorFormats.RGBA) and isinstance(color_reference_list[0], str):
            return [parse_rgb(color) for color in color_reference_list]
        else:
            return color_reference_list
    
    def sample_colors(self, num_samples:int, interpolation_format: ColorFormats = ColorFormats.LAB) -> list[str]:
        """Sample a list of colors from the color reference list, interpolating if necessary
        Args:
            num_samples (int): Number of colors to sample
            interpolation_format (ColorFormats, optional): Colorspace to interpolate through. Defaults to ColorFormats.LAB.
        Returns:
            list[str]: List of sampled colors
        """
        
        #If the number of samples is equal to the number of colors, then just return the colors
        if num_samples == len(self.color_reference_rgb):
            return [rgb_tuple_to_hex(color, upscaled=True) for color in self.color_reference_rgb]

        #Convert the color reference list to the interpolation format
        color_reference_array = np.array([convert_color(color, input_format=self.color_format, output_format=interpolation_format) for color in self.color_reference_rgb])

        #If the interpolator has not been fit, fit it with the color reference list
        if not self.interpolator.is_fitted():
            self.interpolator.fit(color_reference_array, self.weights)

        samples = self.interpolator.sample(num_samples)

        #Turn the samples into a list of colors
        try:
            sampled_colors = [format_to_object(interpolation_format)(*sample, is_upscaled=True) for sample in samples]
        except TypeError:
            sampled_colors = [format_to_object(interpolation_format)(*sample) for sample in samples]

        #Convert the sampled colors to RGB hex
        sampled_colors = [convert_color(color, output_format=format_to_object(ColorFormats.RGB)) for color in sampled_colors]

        #Convert the sampled colors to hex
        sampled_colors = [rgb_tuple_to_hex(color, upscaled=False) for color in sampled_colors]

        #Return the sampled colors
        return sampled_colors
    
    def preview_palette(self, num_samples:int = 10, image_size:tuple[int, int] = (400,200), interpolation_format: ColorFormats = ColorFormats.LAB) -> Image:
        """Preview a palette of colors including colorblind simulations
        Args:
            num_samples (int, optional): Number of sample colors. Defaults to 10.
            image_size (tuple[int, int], optional): Size of the preview image. Defaults to (400,200).
            interpolation_format (ColorFormats, optional): Color format to interpolate through. Defaults to ColorFormats.LAB.
        Returns:
            Image: Preview image of the palette
        """
        #Sample the colors
        sampled_colors = self.sample_colors(num_samples, interpolation_format)

        #Create a new image
        image = Image.new("RGB", image_size)

        #Draw the colors onto the top half of the image
        draw = ImageDraw.Draw(image)

        #Calculate the rectangle dimensions:
        rect_width  = image_size[0] // num_samples
        rect_height = image_size[1] // 5

        #Turn the sampled colors into a list of RGB tuples, then daltonize
        rgb_tuples = np.array([rgb_hex_to_tuple(color) for color in sampled_colors]).reshape((num_samples, 1, 3))
        
        deut_tuples, prot_tuples, trit_tuples = daltonize([rgb_hex_to_tuple(color) for color in sampled_colors])

        #Draw the colors
        for i in range(num_samples):
            #Draw the color without modification
            draw.rectangle((i * rect_width,           0, (i + 1) * rect_width,    rect_height), fill=sampled_colors[i])

            #Grab the daltonized values and turn them into RGB Hex values
            deut_hex        = rgb_tuple_to_hex(deut_tuples[i], upscaled=True)
            prot_hex        = rgb_tuple_to_hex(prot_tuples[i], upscaled=True)
            trit_hex        = rgb_tuple_to_hex(trit_tuples[i], upscaled=True)

            #Generate the luminosity values
            rgb_tuple       = rgb_tuples[i][0]
            grayscale_value = rgb_to_luminosity(rgb_tuple)
            grayscale_hex   = rgb_tuple_to_hex([int(grayscale_value) for _ in range(3)], upscaled=True)

            #Draw the colorblind versions of the color
            draw.rectangle((i * rect_width, 1 * rect_height, (i + 1) * rect_width, 2 * rect_height), fill=deut_hex)
            draw.rectangle((i * rect_width, 2 * rect_height, (i + 1) * rect_width, 3 * rect_height), fill=prot_hex)
            draw.rectangle((i * rect_width, 3 * rect_height, (i + 1) * rect_width, 4 * rect_height), fill=trit_hex)
            draw.rectangle((i * rect_width, 4 * rect_height, (i + 1) * rect_width, 5 * rect_height), fill=grayscale_hex)
        return image

    def validate_palette(self) -> bool:
        """Rudimentary validation, raise warnings if palette is inadequate for color deficient people
        Returns:
            bool: boolean indicating whether the palette is valid
        """
        #Create 50 samples of the colors
        sampled_colors = self.sample_colors(50)

        #Turn the sampled colors into a list of RGB tuples, then daltonize
        rgb_tuples = np.array([rgb_hex_to_tuple(color) for color in sampled_colors]).reshape((50, 1, 3))
        deut_tuples, prot_tuples, trit_tuples = self.daltonize(rgb_tuples)

        #Turn the sampled colors into grayscale
        grayscale_hex = [rgb_to_luminosity_hex(color) for color in sampled_colors]
        #Turn the hex grayscale values into RGB tuples
        grayscale_tuples = [rgb_hex_to_tuple(color) for color in grayscale_hex]

        grayscale_diff = np.diff(grayscale_tuples)
        #Check if the grayscale values are monotonically increasing
    
    def draw_curve(self, num_samples:int):
        """Draw a curve of the palette in RGB colorspace using plotly

        Args:
            num_samples (int): number of samples to draw
        """
        #Sample the colors
        sampled_colors = self.sample_colors(num_samples)

        #Create a new figure
        fig = go.Figure()

        #Turn the sampled colors into a list of RGB tuples, which represent the x, y, and z values
        x, y, z = np.array([rgb_hex_to_tuple(color) for color in sampled_colors]).T

        #Add the scatter plot
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color=sampled_colors)))

        # #Label the axes
        # fig.update_layout(title='Color Palette', xaxis_title='Red', yaxis_title='Green', zaxis_title='Blue')

        #Set limits on the axes
        fig.update_layout(scene=dict(
                                    xaxis=dict(range=[0, 255]), 
                                    yaxis=dict(range=[0, 255]), 
                                    zaxis=dict(range=[0, 255]),
                                    xaxis_title='Red',
                                    yaxis_title='Green',
                                    zaxis_title='Blue'))
        
        #Draw the three background images
        red, green, blue = [], [], []
        for i in range(255):
            for j in range(255):
                red     += [(255, i, j)]
                green   += [(i, 255, j)]
                blue    += [(i, j, 255)]
        
        #Turn the list of tuples into an image
        red_image = Image.new("RGB", (255, 255))
        red_image.putdata(red)
        green_image = Image.new("RGB", (255, 255))
        green_image.putdata(green)
        blue_image = Image.new("RGB", (255, 255))
        blue_image.putdata(blue)

        #Add the images to the figure as the background
        fig.add_layout_image(
            dict(
                source=red_image,
                x=255,
                y=0,

            )
        )

        #Return the figure
        return fig
    
    def update_interpolator(self, interpolator:InterpolatorMixin) -> None:
        """Update the interpolator used to generate colors

        Args:
            interpolator (InterpolatorMixin): interpolator to use
        """
        self.interpolator = interpolator()