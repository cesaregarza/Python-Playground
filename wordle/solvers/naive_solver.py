from .solver_abc import WordleSolverABC
from ..wordle import Wordle
import requests, re
import pandas as pd
import numpy as np

class NaiveSolver(WordleSolverABC):
    def __init__(self, wordle_instance: 'Wordle', word_list:list[str]) -> None:
        self.words_base         = pd.DataFrame(word_list)
        self.initialize(wordle_instance)
    
    def initialize(self, wordle_instance: 'Wordle') -> None:
        self.words              = self.words_base.copy()
        self.game               = wordle_instance
        self.last_guess         = None
        self.last_guess_index   = None
        self.set_statistics()
    
    def set_statistics(self) -> None:
        """Calculates and sets the statistics for the puzzle based on the current state of the puzzle
        """
        #Generate the statistics for the new list of words
        letter_counts, letter_by_columns    = NaiveSolver.generate_statistics(self.words)
        self.letter_counts                  = letter_counts
        self.letter_by_columns              = letter_by_columns
    
    @staticmethod
    def generate_statistics(word_df:pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
        """Generates the statistics for the given list of words

        Args:
            word_df (pd.DataFrame): The list of words

        Returns:
            tuple[dict[str, int], pd.DataFrame]: A tuple containing the letter counts and the letter by columns dataframe
        """

        #Generate a dictionary of the number of times a letter appears in a word
        letter_counts = {
            char: np.any((word_df.values == char), axis=1).sum() 
            for char 
            in 'abcdefghijklmnopqrstuvwxyz'
        }
        
        #Generate a dataframe containing the letter counts on a per-column basis
        word_values = {}
        for col in word_df.columns:
            word_values[col] = word_df[col].value_counts()
        
        word_values_df = pd.DataFrame(word_values)
        return letter_counts, word_values_df
    
    def find_best_word(self) -> pd.Series:
        """Finds the word with the highest likelihood of some letter being correct. Ranks every letter in every position, and returns the word with the lowest rank.

        Returns:
            pd.Series: The word with the highest likelihood of some letter being correct
        """

        
        #Find the word that has the lowest "rank" across all columns
        word_rank_df = self.letter_by_columns.rank(ascending=False)

        #Replace each letter with its rank in the dataframe
        words = self.words.copy()
        for col in words.columns:
            words.loc[:, col] = words.loc[:, col].replace(word_rank_df[col])
        
        #Find the word that has the lowest rank across all columns
        words["sum"]    = words.sum(axis=1)
        best_word_index = words.sort_values(by="sum").iloc[0].name
        return self.words.loc[best_word_index]
    
    def not_in_word_list(self, word:str = None) -> str:
        """Removes the last guess from the list of words and returns the next best guess

        Args:
            word (str, optional): The word to remove from the list of words. If None, the last guess is removed. Defaults to None.

        Returns:
            str: The next best guess
        """
        word        = word if word is not None else ''.join(self.last_guess)
        word_list   = self.words.apply(lambda x: ''.join(x), axis=1)
        words       = self.words.loc[word_list != word]
        self.words  = words
        self.set_statistics()
        return
    
    def generate_guess(self) -> str:
        """Generates the best guess given the puzzle's current state by finding the word with the highest likelihood of some letter being correct

        Returns:
            str: The best guess
        """
        #Find the best word and return it
        self.last_guess         = self.find_best_word()
        self.last_guess_index   = self.last_guess.name
        return ''.join(self.last_guess)
    
    def pass_results(self, results_list:list[int]) -> None:

        words = self.words.copy()

        last_guess_joined = ''.join(self.last_guess)
        
        #Go through the results and generate a minimum number of matches for each letter
        letter_counts = {}
        for i, letter in enumerate(last_guess_joined):
            #Count the number of positive results for each letter
            res = 0 if results_list[i] == 0 else 1
            if letter not in letter_counts:
                letter_counts[letter] = res
            else:
                letter_counts[letter] += res
        
        #Remove all words that have more than the number of correct letters
        for letter, value in letter_counts.items():
            if value == 0:
                #Remove all words that contain a letter that did not match at all
                mask = (words == letter).sum(axis=1) == 0
            elif value < self.last_guess.count():
                #This is a special case where the amount of matches is less than the amount of letters in the self.last_guess
                #This mask removes all words that do not contain exactly the amount of matches
                mask = (words == letter).sum(axis=1) == value
            else:
                #Otherwise, remove all words that contain less than the amount of matches
                mask = (words == letter).sum(axis=1) >= value
            words = words[mask]
        
        #Go through the results_list that have correct position and filter accordingly
        for i, result in enumerate(results_list):
            if result == 1:
                mask   = (words[i] != self.last_guess[i])
                words   = words[mask]
            elif result == 2:
                mask    = (words[i] == self.last_guess[i])
                words   = words[mask]
        
        self.words = words
        self.set_statistics()
        return