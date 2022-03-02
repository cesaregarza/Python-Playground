import numba as nb
import numpy as np

@nb.jit(nb.float64(nb.int64[:], nb.int64[:]), nopython=True)
def compute_entropy(row: np.ndarray, possible_solutions:np.ndarray) -> np.ndarray:
    """Private method to compute the entropy of the ternary matrix.

    Args:
        row (np.ndarray): Array of ternary values representing a row of the ternary matrix.
        possible_solutions (np.ndarray): Array of ternary values representing the possible solutions.

    Returns:
        np.ndarray: Array of entropies for each row of the matrix.
    """
    #Mask the row to only include the possible solutions
    row = row[np.where(possible_solutions)]
    #Compute the probability of each row through bincount
    row_probabilities = np.bincount(row) / possible_solutions.sum()
    #Mask out values with 0
    row_probabilities = row_probabilities[row_probabilities > 0]
    #Compute the log of the probabilities
    log_row_probabilities = np.log2(row_probabilities)
    #Compute the entropy of each row
    return -np.sum(row_probabilities * log_row_probabilities)

@nb.jit(nb.float64[:](nb.int64[:, :], nb.int64[:]), nopython=True, parallel=True)
def compute_entropy_all(matrix:np.ndarray, possible_solutions:np.ndarray) -> np.ndarray:
    """Compute the entropy of all words in the word matrix.

    Args:
        matrix (np.ndarray): The word matrix, with values being ternary representations of the results.
        possible_solutions (np.ndarray): The possible solutions index. 0 means exclude, 1 means include.

    Returns:
        np.ndarray: The entropy of all words in the word matrix.
    """
    return_array    = np.zeros(matrix.shape[0])
    columns         = np.where(possible_solutions)
    for idx in nb.prange(matrix.shape[0]):
        if possible_solutions[idx] == 0:
            continue
        row                     = matrix[idx]
        row                     = row[columns]
        row_probabilities       = np.bincount(row) / possible_solutions.sum()
        row_probabilities       = row_probabilities[row_probabilities > 0]
        log_row_probabilities   = np.log2(row_probabilities)
        return_array[idx]       = -np.dot(row_probabilities, log_row_probabilities)
    return return_array

@nb.jit(nb.int64[:](nb.int64[:], nb.int64, nb.int64[:]))
def compute_possible_solutions(row:np.ndarray, value:int, possible_solutions:np.ndarray) -> np.ndarray:
    """Compute the possible solutions for a row.

    Args:
        row (np.ndarray): The row of the ternary matrix.
        value (int): The value to compare the row to.

    Returns:
        np.ndarray: The possible solutions for the row.
    """
    return np.where(row == value, 1, 0) & possible_solutions