from typing import Optional
from .solvers.boilerplate import BoilerplateWordleSolver
from .wordle import Wordle
import pandas as pd
import numpy as np
from numpy import log2 as log

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
            self.word_matrix.columns.name   = "solution"
        else:
            self.word_matrix = self.compute_word_matrix(allowed_word_list, solution_word_list)

    def initialize(self, wordle_instance: 'Wordle') -> None:
        self.game       = wordle_instance
        self.word_list  = [*self.base_allowed]

    def compute_entropy(self, word:str) -> int:
        word_distribution = self.word_matrix.loc[word, :].value_counts()

        probability = word_distribution / word_distribution.sum()
        entropy = -np.sum(probability * log(probability))
        return entropy
    
    def generate_guess(self):
        pass

    def submit_guess(self, guess:str) -> None:
        pass

    def pass_results(self, results:list[str]) -> None:
        pass