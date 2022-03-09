from itertools import count
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
    row                     = row[np.where(possible_solutions)]
    #Compute the probability of each row through bincount
    row_probabilities       = np.bincount(row) / possible_solutions.sum()
    #Mask out values with 0
    row_probabilities       = row_probabilities[row_probabilities > 0]
    #Compute the log of the probabilities
    log_row_probabilities   = np.log2(row_probabilities)
    #Compute the entropy of each row
    return -np.dot(row_probabilities, log_row_probabilities)

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
        row                     = matrix[idx]
        row                     = row[columns]
        row_probabilities       = np.bincount(row) / possible_solutions.sum()
        row_probabilities       = row_probabilities[row_probabilities > 0]
        log_row_probabilities   = np.log2(row_probabilities)
        return_array[idx]       = -np.dot(row_probabilities, log_row_probabilities)
    return return_array

@nb.jit(nb.float64[:](nb.int64[:, :], nb.int64[:]), nopython=True, parallel=True)
def compute_entropy_all_two_step(matrix:np.ndarray, possible_solutions:np.ndarray) -> np.ndarray:
    """Compute the entropy of all words in the word matrix, but with two steps

    Args:
        matrix (np.ndarray): The word matrix, with values being ternary representations of the results.
        possible_solutions (np.ndarray): The possible solutions index. 0 means exclude, 1 means include.

    Returns:
        np.ndarray: The entropy of all words in the word matrix.
    """
    return_array    = np.zeros(matrix.shape[0])
    columns         = np.where(possible_solutions)

    #Calculate the two-step entropy for every single word
    for idx in nb.prange(matrix.shape[0]):
        #Extract the row, mask it to only include the possible solutions, and compute the entropy using np.bincount
        row                     = matrix[idx]
        row                     = row[columns]
        bincount                = np.bincount(row)
        row_probabilities       = bincount / possible_solutions.sum()
        row_probabilities       = row_probabilities[row_probabilities > 0]
        log_row_probabilities   = np.log2(row_probabilities)
        entropy                 = -np.dot(row_probabilities, log_row_probabilities)
        max_second_entropy      = 0
        
        #Calculate the second step entropy.
        for possible_result, count in enumerate(bincount):
            #If there were no counts for this index, skip it
            if count == 0:
                continue
            #Find the columns that would be possible if this possible result is returned
            possible_columns = np.where(row == possible_result, 1, 0)
            
            #Compute the entropy of the possible columns
            for idx2 in np.arange(matrix.shape[0]):
                if possible_solutions[idx2] == 0:
                    continue
                row2                    = matrix[idx2]
                row2                    = row2[possible_columns]
                row2_probabilities      = np.bincount(row2) / possible_solutions.sum()
                row2_probabilities      = row2_probabilities[row2_probabilities > 0]
                log_row2_probabilities  = np.log2(row2_probabilities)
                second_entropy          = -np.dot(row2_probabilities, log_row2_probabilities)
                if second_entropy > max_second_entropy:
                    max_second_entropy = second_entropy

        #The entropy of the row is the max of the first and second step entropies
        return_array[idx] = entropy + max_second_entropy
    
    return return_array

@nb.jit(nb.float64[:](nb.int64[:, :], nb.int64[:], nb.int64, nb.float64[:]), nopython=True, parallel=True)
def compute_entropy_second_step(matrix:np.ndarray, 
                                possible_solutions:np.ndarray, 
                                top_k_values:int,
                                first_step_entropies:np.ndarray) -> np.ndarray:
    return_array    = np.zeros(matrix.shape[0])
    columns         = np.where(possible_solutions)

    #Find the top k indices
    top_k_indices = np.argsort(first_step_entropies)[-top_k_values:]

    #Calculate the second step entropy for all top k words
    for idx in nb.prange(top_k_indices.shape[0]):
        k_idx = top_k_indices[idx]
        if possible_solutions[k_idx] == 0:
            continue

        row                     = matrix[k_idx]
        row_probabilities       = np.bincount(row[columns]) / possible_solutions.sum()
        entropies               = np.zeros(matrix.shape[0])
        #Go through all possible results and recompute the entropy
        for possible_result, prob in enumerate(row_probabilities):
            if count == 0:
                continue

            #Find the columns that would be possible if this possible result is returned
            possible_columns = np.where(row == possible_result, 1, 0) & possible_solutions
            columns2         = np.where(possible_columns)

            expected_entropy = 0
            #Compute the entropy of the possible columns
            for idx2 in np.arange(top_k_indices.shape[0]):
                k_idx2 = top_k_indices[idx2]

                row2                    = matrix[k_idx2]
                row2                    = row2[columns2]
                row2_probabilities      = np.bincount(row2) / possible_columns.sum()
                row2_probabilities      = row2_probabilities[row2_probabilities > 0]
                log_row2_probabilities  = np.log2(row2_probabilities)
                second_entropy          = -np.dot(row2_probabilities, log_row2_probabilities)
                
                expected_entropy += second_entropy * prob
            
            entropies[k_idx] = expected_entropy
            
        return_array[k_idx] = first_step_entropies[k_idx] + entropies.max()

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