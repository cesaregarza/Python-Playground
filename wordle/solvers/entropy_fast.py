from typing import Optional

from sqlalchemy import column

from .boilerplate import BoilerplateWordleSolver
from ..wordle import Wordle
import pandas as pd
import numpy as np
from numpy import log2 as log
import numba as nb
from functools import partial

class FastEntropicSolver(BoilerplateWordleSolver):
    """Entropy-based Wordle Solver. This version of the entropic solver trades readability for speed."""

    def __init__(self, allowed_word_list:list[str],
                       solution_word_list:Optional[list[str]] = None,
                       precomputed_word_matrix_path:Optional[str] = None) -> None:
        
        #If a solution word list is provided, separate it from the allowed word list.
        if solution_word_list is None:
            solution_word_list = [*allowed_word_list]
        
        if precomputed_word_matrix_path:
            self.word_matrix                = pd.read_csv(precomputed_word_matrix_path, index_col=0)
            self.word_matrix.columns.name   = "solution_word"
            self.word_matrix.index.name     = "guess_word"
            #If the word "guess" is in the word matrix as a column, it will have been changed to "guess.1", so we need 
            #to fix it.
            if "guess.1" in self.word_matrix.columns:
                self.word_matrix.loc[:, "guess"]    = self.word_matrix.loc[:, "guess.1"]
                self.word_matrix                    = self.word_matrix.drop(columns=["guess.1"])
        else:
            self.word_matrix = self.compute_word_matrix(allowed_word_list, solution_word_list)
        
        #Replace the word matrix string values with their ternary equivalents
        self.ternary = partial(int, base=3)
    
    def initialize(self, wordle_instance: 'Wordle' = None) -> None:
        self.game_matrix:pd.DataFrame    = None
    
    @staticmethod
    @nb.jit(nb.float64(nb.int64[:]), nopython=True)
    def _compute_entropy(row: np.ndarray) -> np.ndarray:
        """Private method to compute the entropy of the ternary matrix.

        Args:
            row (np.ndarray): Array of ternary values representing a row of the ternary matrix.

        Returns:
            np.ndarray: Array of entropies for each row of the matrix.
        """
        #Compute the probability of each row through bincount
        row_probabilities = np.bincount(row) / row.shape[0]
        #Mask out values with 0
        row_probabilities = row_probabilities[row_probabilities > 0]
        #Compute the log of the probabilities
        log_row_probabilities = np.log2(row_probabilities)
        #Compute the entropy of each row
        return -np.sum(row_probabilities * log_row_probabilities)
    
    def compute_entropy_all(self) -> np.ndarray:
        """Compute the entropy of all words in the word matrix.

        Returns:
            np.ndarray: The entropy of all words in the word matrix, sorted by entropy.
        """
        matrix = self.word_matrix if self.game_matrix is None else self.game_matrix
        return np.apply_along_axis(self._compute_entropy, 1, matrix.values)
    
    def compute_best_guess(self) -> str:
        """Compute the best guess

        Returns:
            str: The best guess via entropy
        """
        values = self.compute_entropy_all()
        matrix = self.word_matrix if self.game_matrix is None else self.game_matrix
        return matrix.index[values.argmax()]

class FastOneStepEntropicSolver(FastEntropicSolver):
    def __init__(self, allowed_word_list:list[str],
                       solution_word_list:Optional[list[str]] = None,
                       precomputed_word_matrix_path:Optional[str] = None) -> None:
        super().__init__(allowed_word_list, solution_word_list, precomputed_word_matrix_path)
        self.game_matrix        = None
        self.best_first_guess   = self.compute_best_guess()

    def initialize(self, wordle_instance: 'Wordle') -> None:
        self.last_guess = None
        return super().initialize(wordle_instance)

    def generate_guess(self):
        if self.last_guess is None:
            self.last_guess = self.best_first_guess
            return self.last_guess
        self.last_guess = self.compute_best_guess()
        return self.last_guess
    
    def pass_results(self, results:list[str]) -> None:
        
        results_value       = self.ternary("".join([str(x) for x in results]))
        matrix              = self.game_matrix if self.game_matrix is not None else self.word_matrix
        last_guess_series   = matrix.loc[self.last_guess]
        columns_possible    = last_guess_series.loc[last_guess_series == results_value].index
        self.game_matrix    = matrix.loc[columns_possible, columns_possible]
        return