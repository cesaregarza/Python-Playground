from .solver_abc import WordleSolverABC
from ..wordle import Wordle

from typing import Optional
import os, pathlib, hashlib
from multiprocessing import Pool
import pandas as pd
import numpy as np

class BoilerplateWordleSolver(WordleSolverABC):
    """Boilerplate code for a solver. Includes helpful functions for a solver to use."""

    @staticmethod
    def compute_word_matrix(allowed_word_list:list[str],
                            solution_word_list:Optional[list[str]] = None,
                            cores:Optional[int] = None) -> pd.DataFrame:
        """Compute the word matrix for the given word list. This returns a pandas DataFrame where the columns are the
        solution words, the rows are the guess words, and the values are the results of the evaluation of the guess word
        against the solution word. This is functionally a wrapper for compute_sub_matrix, but is split out to allow for 
        parallelization. If the solution_word_list is not provided, the columns will include all words in 
        allowed_word_list. If solution_word_list is provided, the columns will be limited to the solution_word_list.

        Args:
            allowed_word_list: The list of words that are allowed as guesses. Will combine with the solution_word_list 
            regardless of whether it is provided to assure the rows are complete.
            solution_word_list: The list of words that are allowed as solutions. If not provided, the columns will be 
            limited to the allowed_word_list instead.
            cores: The number of cores to use for parallelization. If not provided, will use the number of cores on the 
            system.
        
        Returns:
            A pandas DataFrame where the columns are the solution words, the rows are the guess words, and the values are
            the results of the evaluation of the guess word against the solution word.
        """
        cores = os.cpu_count() if cores is None else cores

        word_length = len(allowed_word_list[0])

        #Combine the allowed_word_list and the solution_word_list
        if solution_word_list is None:
            solution_word_list = allowed_word_list
        else:
            allowed_word_list = list(set(allowed_word_list + solution_word_list))

        #Sort the word lists
        allowed_word_list.sort()
        solution_word_list.sort()

        #Generate a list of all word pairs. The wordle evaluation is a commutative operation and generating all scores
        #is a cpu-intensive operation. As such, we will minimize the number of word pairs by only generating unique pairs
        #and then combining them into a single matrix. A word against itself will always have the same score, so will be
        #ignored from this generation.
        
        #Solutions are not commutative, so all solutions must be generated.
        word_list = [
            (solution_word_list[i], solution_word_list[j])
            for i in range(len(solution_word_list))
            for j in range(len(solution_word_list))
        ]

        #Allowed words are treated differently, they must be paired with all other words in the solution_word_list. This
        #is because while allowed vs solution is possible, solution vs allowed is not.
        allowed_not_solution = list(set(allowed_word_list) - set(solution_word_list))
        if len(allowed_not_solution) > 0:
            word_list += [
                (allowed_word, solution_word)
                for allowed_word in allowed_not_solution
                for solution_word in solution_word_list
            ]
        del allowed_not_solution

        #Columns will be passed to the sub-matrix as-is, but rows will be divided up into even-ish chunks
        row_length = int(np.ceil(len(word_list) / cores))
        rows       = [
            word_list[i * row_length : (i + 1) * row_length]
            for i in range(cores)
        ]

        #Use multiprocessing to compute each given chunk of rows to maximize parallelization
        submat_func = BoilerplateWordleSolver.compute_sub_matrix
        with Pool(cores) as pool:
            word_series = pd.concat(pool.map(submat_func, rows))

        #Unstack the series into a dataframe, and sort both rows and columns
        word_matrix = word_series.unstack()
        return word_matrix.sort_index().sort_index(axis=1)
    
    @staticmethod
    def compute_sub_matrix(word_list:list[tuple[str, str]]) -> pd.Series:
        """Compute the word matrix for the given word list. This returns a pandas Series with the first index being the
        guess word, and the second index being the solution word. The values are the results of the evaluation of the guess.
        This format allows for unstacking the series into a dataframe. This is a helper function for compute_word_matrix.

        Args:
            word_list (list[tuple[str, str]]): The list of word pairs to compute the word matrix for.

        Returns:
            pd.Series: The series of word matrix scores.
        """
        
        #Generate the list of results for each given word pair
        data = [
            BoilerplateWordleSolver.evaluate_word_str(guess, solution)
            for guess, solution in word_list
        ]
        #Generate the reverse list of results for each given word pair and create an index containing both forward and
        #reverse pairs
        index = pd.MultiIndex.from_tuples(word_list, names=['guess', 'solution'])

        #Generate the series using the data twice, once for the forward and once for the reverse.
        return pd.Series(data, index=index)
    
    @staticmethod
    def evaluate_word_str(guess:str, solution:str) -> str:
        """Evaluate the given guess against the given solution, and return the result as a string. This is a wrapper for
        the wordle evaluation function.

        Args:
            guess (str): The guess word.
            solution (str): The solution word.

        Returns:
            str: The result of the evaluation. 2 for a full match, 1 for a letter match, and 0 for no match.
        """
        result = Wordle.evaluate_word(solution, guess)
        result = [str(x) for x in result]
        return "".join(result)
    
    @staticmethod
    def check_for_precomputed(file_path:Optional[str] = None) -> bool:
        """Check for the existence of a precomputed word matrix at the given file path.

        Args:
            file_path (str): The file path to check for the precomputed word matrix.

        Returns:
            bool: True if the precomputed word matrix exists, False otherwise.
        """
        if file_path is not None:
            return os.path.isfile(file_path)
        
        #If no file path is given, check for the default file path
        default_path = pathlib.Path(__file__).parent.parent / "precompute" / "word_matrix.csv"
        return os.path.isfile(default_path)