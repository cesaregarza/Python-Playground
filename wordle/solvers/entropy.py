from typing import Optional

from .boilerplate import BoilerplateWordleSolver
from ..wordle import Wordle
import pandas as pd
import numpy as np
from numpy import log2 as log
import numba as nb

class EntropicSolver(BoilerplateWordleSolver):

    def __init__(self, allowed_word_list:list[str],
                       solution_word_list:Optional[list[str]] = None,
                       precomputed_word_matrix_path:Optional[str] = None) -> None:
        
        if solution_word_list is None:
            self.base_solution  = [*allowed_word_list]
            self.base_allowed   = [*allowed_word_list]
        else:
            self.base_solution  = [*solution_word_list]
            self.base_allowed   = list(set(allowed_word_list + solution_word_list))
        
        self.num_words      = len(self.base_allowed)
        #If a precomputed word matrix is provided, use it. Otherwise, compute it.
        if precomputed_word_matrix_path:
            self.word_matrix                = pd.read_csv(precomputed_word_matrix_path, dtype=str, index_col=0)
            self.word_matrix.columns.name   = "solution_word"
            self.word_matrix.index.name     = "guess_word"
            #If the word "guess" is in the word matrix as a column, it will have been changed to "guess.1", so we need 
            #to fix it.
            if "guess.1" in self.word_matrix.columns:
                self.word_matrix.loc[:, "guess"]    = self.word_matrix.loc[:, "guess.1"]
                self.word_matrix                    = self.word_matrix.drop(columns=["guess.1"])
        else:
            self.word_matrix = self.compute_word_matrix(allowed_word_list, solution_word_list)

    def initialize(self, wordle_instance: 'Wordle' = None) -> None:
        self.game_matrix:pd.DataFrame    = None
    
    @staticmethod
    def __compute_entropy(row:pd.Series) -> float:
        """Private method to compute the entropy of a row. Designed to be used with pandas apply.

        Args:
            row (pd.Series): A row of the word matrix.

        Returns:
            float: The entropy of the row.
        """
        probability   = row.value_counts(normalize=True).values
        entropy       = EntropicSolver.__compute_entropy_probability(probability)
        return entropy
    
    @staticmethod
    @nb.jit(nb.float64(nb.float64[:]), nopython=True)
    def __compute_entropy_probability(probability:np.ndarray) -> float:
        """Private method to compute the entropy of a row. Designed to be used with numba.

        Args:
            probability (np.ndarray): A row of the word matrix.

        Returns:
            float: The entropy of the row.
        """
        entropy       = -np.sum(probability * np.log2(probability))
        return entropy

    def compute_entropy(self, word:str) -> float:
        """Compute the entropy of a single word.

        Args:
            word (str): The word to compute the entropy of.

        Returns:
            float: The entropy of the word.
        """
        if self.game_matrix is None:
            return self.word_matrix.loc[word].pipe(self.__compute_entropy)
        return self.game_matrix.loc[word].pipe(self.__compute_entropy)
    
    def compute_entropy_all(self) -> pd.Series:
        """Compute the entropy of all words in the word matrix.

        Returns:
            pd.Series: The entropy of all words in the word matrix, sorted by entropy.
        """
        if self.game_matrix is None:
            return self.word_matrix.apply(self.__compute_entropy, axis=1)
        return self.game_matrix.apply(self.__compute_entropy, axis=1)
    
    def compute_best_guess(self) -> str:
        """Compute the best first guess.

        Returns:
            str: The best first guess.
        """
        entropy_all = self.compute_entropy_all()
        return entropy_all.idxmax()
    
    def generate_guess(self):
        pass

    def pass_results(self, results:list[str]) -> None:
        pass

class OneStepEntropicSolver(EntropicSolver):
    def __init__(self, allowed_word_list:list[str],
                       solution_word_list:Optional[list[str]] = None,
                       precomputed_word_matrix_path:Optional[str] = None) -> None:
        super().__init__(allowed_word_list, solution_word_list, precomputed_word_matrix_path)
        self.game_matrix        = self.word_matrix.copy()
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
        
        results_string      = "".join([str(x) for x in results])
        word_matrix         = self.game_matrix if self.game_matrix is not None else self.word_matrix
        last_guess_series   = word_matrix.loc[self.last_guess]
        columns_possible    = last_guess_series.loc[last_guess_series == results_string].index
        self.game_matrix    = word_matrix.loc[columns_possible, columns_possible]
        return