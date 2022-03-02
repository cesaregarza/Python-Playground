from typing import Optional

from .boilerplate import BoilerplateWordleSolver
from ..wordle import Wordle
import pandas as pd
import numpy as np
from numpy import log2 as log
import numba as nb
from functools import partial
from .compiled_functions import compute_entropy, compute_entropy_all, compute_possible_solutions

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
                self.word_matrix = self.word_matrix.rename(columns={"guess.1":"guess"})
            
            self.word_matrix = self.word_matrix.sort_index(axis=0).sort_index(axis=1)
        else:
            self.word_matrix = self.compute_word_matrix(allowed_word_list, solution_word_list)
        
        #Replace the word matrix string values with their ternary equivalents
        self.ternary        = partial(int, base=3)
        #Store the rows and columns of the word matrix
        self.row_index      = {i: word for i, word in enumerate(self.word_matrix.index)}
        self.row_reverse    = {word: i for i, word in self.row_index.items()}
        self.col_index      = {i: word for i, word in enumerate(self.word_matrix.columns)}
        self.col_reverse    = {word: i for i, word in self.col_index.items()}
        self.val_matrix     = self.word_matrix.values
    
    def initialize(self, wordle_instance: 'Wordle' = None) -> None:
        self.possible_solutions:list[int] = np.ones(self.word_matrix.shape[1], dtype=np.int64)

    def compute_entropy(self, word:str) -> float:
        """Compute the entropy of a single word.

        Args:
            word (str): The word to compute the entropy of.

        Returns:
            float: The entropy of the word.
        """
        index = self.row_rev[word]
        return compute_entropy(self.val_matrix[index], self.possible_solutions)
    
    def compute_entropy_all(self) -> np.ndarray:
        return compute_entropy_all(self.val_matrix, self.possible_solutions)

    def compute_best_guess(self) -> int:
        """Compute the best guess

        Returns:
            str: The best guess via entropy
        """
        values = self.compute_entropy_all()
        return np.argmax(values)
    
    def row_index_to_word(self, index:int) -> str:
        """Convert an index to a word.

        Args:
            index (int): The index to convert.

        Returns:
            str: The word corresponding to the index.
        """
        return self.row_index[index]
    
    def col_index_to_word(self, index:int) -> str:
        """Convert an index to a word.

        Args:
            index (int): The index to convert.

        Returns:
            str: The word corresponding to the index.
        """
        return self.col_index[index]
    
    def compute_best_word(self) -> str:
        """Compute the best word

        Returns:
            str: The best word via entropy
        """
        return self.row_index[self.compute_best_guess()]

class FastOneStepEntropicSolver(FastEntropicSolver):
    def __init__(self, allowed_word_list:list[str],
                       solution_word_list:Optional[list[str]] = None,
                       precomputed_word_matrix_path:Optional[str] = None) -> None:
        super().__init__(allowed_word_list, solution_word_list, precomputed_word_matrix_path)
        self.possible_solutions = np.ones(self.word_matrix.shape[1], dtype=np.int64)
        self.best_first_guess   = self.compute_best_guess()

    def initialize(self, wordle_instance: Optional['Wordle'] = None) -> None:
        self.last_guess = None
        return super().initialize(wordle_instance)

    def generate_guess(self) -> str:
        """Generate the best guess.

        Returns:
            str: The best guess via maximum entropy.
        """
        if self.last_guess is None:
            self.last_guess = self.best_first_guess
        elif self.possible_solutions.sum() > 1:
            self.last_guess = self.compute_best_guess()
        else:
            self.last_guess = np.argmax(self.possible_solutions)
        return self.row_index_to_word(self.last_guess)
    
    def pass_results(self, results:list[str]) -> None:
        """Feed the results of the last guess back into the solver.

        Args:
            results (list[str]): The results of the last guess.
        """
        
        results_value           = self.ternary("".join([str(x) for x in results]))
        row                     = self.val_matrix[self.last_guess]
        self.possible_solutions = compute_possible_solutions(row, results_value, self.possible_solutions)
        return
    
    feed_results = pass_results