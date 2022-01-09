import numpy.typing as npt
import numpy as np

def wedge(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    """Returns the wedge product of two vectors.

    Args:
        a (ArrayLike): A vector.
        b (ArrayLike): A vector.

    Returns:
        np.ndarray: The wedge product of a and b, c, as a list c_ij, when j > i from i = 0 to j = n. e.g. for 3 dimensional cartesian vectors, you'd get [c_xy, c_xz, c_yz] which is equivalent to [z, y, -x].
    """
    #Turn both inputs into numpy arrays, if they aren't already.
    a,b = np.array(a), np.array(b)

    #Find the outer product, which will always be square.
    outer = np.outer(a, b)

    #Subtract the outer product from its transpose.
    diff = outer - outer.T

    #Return the result.
    return diff[np.triu_indices_from(diff, k=1)]

#Alternate name for wedge product.
exterior = wedge

def inverse_triangular(n:int, rounding_places:int = 3) -> float:
    """Returns the inverse of the triangular number n.

    Args:
        n (int): An integer.
        rounding_places (int): The number of decimal places to round to.

    Returns:
        float: The inverse of the triangular number n.
    """
    #Return the inverse of the triangular number.
    return np.round(np.sqrt(2 * n + (1 / 4)) - (1 / 2), rounding_places)