# %%
import requests, re
import pandas as pd
import numpy as np

class WordleSolver:

    def __init__(self, word_length:int = 5) -> None:
        
        #Retrieve the list of k-letter words through the method get_words()
        self.word_length                    = word_length
        self.words                          = self.get_words()
        self.last_guess                     = None
        self.last_guess_index               = None
        self.set_statistics()
    
    def get_words(self) -> pd.DataFrame:
        """Retrieves a list of words and creates a dataframe with them.

        Returns:
            pd.DataFrame: A dataframe with the words
        """
        #Retrieve the list of words from the website below
        url         = 'https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt'
        response    = requests.get(url)

        #Split the words into a list, and clean the list by removing non-alphabetical characters
        words       = response.text.split('\n')
        words       = [re.sub(r"[^a-zA-Z]", "", word) for word in words]

        #Filter the list to only include words with 5 letters, and split each word into a list of letters
        words       = [list(word) for word in words if len(word) == self.word_length]

        #Turn the word list into a dataframe where each letter is a column
        word_df    = pd.DataFrame(words)
        return word_df
    
    def set_statistics(self) -> None:
        """Calculates and sets the statistics for the puzzle based on the current state of the puzzle
        """
        #Generate the statistics for the new list of words
        letter_counts, letter_by_columns    = WordleSolver.generate_statistics(self.words)
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
        return self.guess_word()

    
    def guess_word(self) -> str:
        """Generates the best guess given the puzzle's current state by finding the word with the highest likelihood of some letter being correct

        Returns:
            str: The best guess
        """
        #Find the best word and return it
        self.last_guess         = self.find_best_word()
        self.last_guess_index   = self.last_guess.name
        return ''.join(self.last_guess)
    
    def solve(self, results:list[int] = None, input_word:str = None) -> str:
        """Solver for the Wordle puzzle. Each instance of the solve method will return a word that fits the results

        Args:
            results (list[int], optional): A list containing the different positional results of the puzzle. 0 is a miss, 1 is correct letter in wrong position, 2 is correct letter in correct position. Defaults to None.
            input_word (str, optional): If provided, the solver will use this word instead of the previous best guess. Defaults to None.

        Raises:
            ValueError: If the results list is not provided, or if the results list is not the correct length.

        Returns:
            str: The next best guess for the puzzle.
        """
        #Filter the list of words based on the results.
        #0 means that the letter is not in the word, 1 means that the letter is in the word but not in the correct position
        #2 means that the letter is in the word and in the correct position
        if input_word is not None:
            guess = input_word
        else:
            guess = self.last_guess
        if guess is None:
            return self.guess_word()
        elif (results is None) or len(results) != self.word_length:
            raise ValueError("Results must be provided on subsequent guesses")
        
        guess = list(guess)

        words = self.words.copy()
        
        #Go through the results and generate a minimum number of matches for each letter
        letter_counts = {}
        for i, letter in enumerate(guess):
            #Count the number of positive results for each letter
            res = 0 if results[i] == 0 else 1
            if letter not in letter_counts:
                letter_counts[letter] = res
            else:
                letter_counts[letter] += res
        
        #Remove all words that have more than the number of correct letters
        for letter, value in letter_counts.items():
            if value == 0:
                #Remove all words that contain a letter that did not match at all
                mask = (words == letter).sum(axis=1) == 0
            elif value < guess.count(letter):
                #This is a special case where the amount of matches is less than the amount of letters in the guess
                #This mask removes all words that do not contain exactly the amount of matches
                mask = (words == letter).sum(axis=1) == value
            else:
                #Otherwise, remove all words that contain less than the amount of matches
                mask = (words == letter).sum(axis=1) >= value
            words = words[mask]
        
        #Go through the results that have correct position and filter accordingly
        for i, result in enumerate(results):
            if result == 1:
                mask   = (words[i] != guess[i])
                words   = words[mask]
            elif result == 2:
                mask    = (words[i] == guess[i])
                words   = words[mask]
        
        #Set the new list of words
        self.words = words
        self.set_statistics()
        return self.guess_word()

# class WordleSolution:
#     def __init__(self, solution_word:str):
#         self.solution_word = solution_word
#         self.letter_counts = {letter: list(solution_word).count(letter) for letter in 'abcdefghijklmnopqrstuvwxyz'}

#     def verify_guess(self, guess:str) -> list[int]:
#         return_list = [0] * len(guess)
#         letter_counts = {**self.letter_counts}
#         for i, letter in enumerate(guess):
#             if letter == self.solution_word[i]:
#                 return_list[i] = 2
#                 letter_counts[letter] -= 1
        
#         for i, letter in enumerate(guess):
#             #If the letter is in the guess more times than in the solution, only count the first letter as a match
#             if (letter_counts[letter] > 0) and (return_list[i] == 0):
#                 return_list[i] = 1
#                 letter_counts[letter] -= 1
        
#         return return_list
# # %%
# from tqdm import tqdm
# #Testing
# wordle          = WordleSolver()
# word_list       = wordle.words.apply(lambda x: ''.join(x), axis=1)
# attempt_list    = []
# for solution in tqdm(word_list.sample(100)):
#     wordle          = WordleSolver()
#     wordle_solution = WordleSolution(solution)
#     attempts = 1
#     guess = wordle.solve()
#     res = wordle_solution.verify_guess(guess)
#     results = [(guess, res)]
#     while res != [2, 2, 2, 2, 2]:
#         try:
#             guess = wordle.solve(res)
#         except IndexError as e:
#             print(f"Solution: {solution}")
#             for row in results:
#                 print(f"Guess: {row[0]}, Result: {row[1]}")
#             raise IndexError(e)
#         res = wordle_solution.verify_guess(guess)
#         attempts += 1
#         results += [(guess, res)]

#     attempt_list += [(solution, attempts)]

# # %%
# wordle = WordleSolver()
# ws = WordleSolution('enmew')
# ws.verify_guess('sores')
# %%
