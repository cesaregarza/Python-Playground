import random

class GameOverException(Exception):
    pass

class NotInWordListException(Exception):
    pass

class HardModeException(Exception):
    pass

class Wordle:

    def __init__(self, allowed_word_list:   list[str], 
                       solution_word_list:  list[str], 
                       max_guesses:         int,
                       hard_mode:           bool = False,
                       **kwargs) -> None:
        
        #Deduplicate the lists
        self.solution_word_list = list(set(solution_word_list))
        self.allowed_word_list  = list(set(allowed_word_list).union(set(solution_word_list)))

        #Figure out word length
        self.word_length = len(self.solution_word_list[0])

        #Validate that each word in the solution list is the same length
        for word in self.allowed_word_list:
            if len(word) != self.word_length:
                raise ValueError(f"Every word in the solution list must be the same length, but {word} is {len(word)} "
                                 f"characters long instead of {self.word_length}")

        self.max_guesses                        = max_guesses
        self.hard_mode                          = hard_mode
        self.game_running:bool                  = False
        self.guesses:list[str]                  = []
        self.scores:list[tuple[str,list[int]]]  = []
    
    def game_method(func):
        def wrapper(self:'Wordle', *args, **kwargs):
            if not self.game_running:
                raise Exception("Game must be started before calling this method")
            else:
                return func(self, *args, **kwargs)
        return wrapper
    
    def start_game(self) -> None:
        self.game_running   = True
        self.guesses        = []
        self.__solution_word  = random.choice(self.solution_word_list)
    
    def start_game_select_word(self, solution_word: str) -> None:
        if solution_word not in self.solution_word_list:
            raise ValueError(f"{solution_word} is not in the solution list")
        self.game_running       = True
        self.guesses            = []
        self.__solution_word    = solution_word
    
    @staticmethod
    def evaluate_word(solution: str, guess: str) -> list[int]:
        """Evaluates the guess and returns a list indicating each letter's score. \n
         - 0: Letter not in the solution word
         - 1: Letter in the solution word, but not in the correct position
         - 2: Letter in the solution word, and in the correct position

        Args:
            solution (str): The solution word
            guess (str): The guess to evaluate

        Returns:
            list[int]: A list of scores for each letter in the guess
        """
        guess_list  = list(guess)
        return_list = [0 for _ in range(len(solution))]

        #Correct letter and position
        for position, letter in enumerate(guess_list):
            if solution[position] == letter:
                return_list[position] = 2
        
        #Remove correct letters and positions from the guess list and solution list
        guess_list      = [(letter, position) for position, letter in enumerate(guess_list) if return_list[position] != 2]
        solution_list   = [letter for position, letter in enumerate(solution) if return_list[position] != 2]
        #Remaining letters
        while len(guess_list):
            letter, position = guess_list.pop(0)
            if letter in solution_list:
                return_list[position] = 1
                solution_list.remove(letter)

        return return_list
    
    @game_method
    def __evaluate_guess(self, guess: str) -> list[int]:
        return self.evaluate_word(self.__solution_word, guess)
    
    @game_method
    def submit_guess(self, guess: str) -> list[int]:

        #Check if the guess is valid
        if len(guess) != self.word_length:
            raise ValueError(f"Guess must be {self.word_length} characters long")
        
        #Check if the guess is in the allowed word list
        if guess not in self.allowed_word_list:
            raise NotInWordListException(f"Guess must be in the allowed word list")
        
        #If in hard mode, make sure any letters previously guessed correctly are included in the guess
        if self.hard_mode and len(self.guesses) > 0:
            if not self.__validate_hard_mode(guess):
                raise HardModeException(f"Guess must contain all letters previously guessed correctly")
        
        if len(self.guesses) == self.max_guesses:
            raise GameOverException("Number of allowed guesses exceeded")
        
        #Evaluate the guess
        score = self.__evaluate_guess(guess)

        if score == [2] * self.word_length:
            self.game_running = False
        else:
            self.guesses += [guess]
            self.scores += [(guess, score)]
        return score
    
    @game_method
    def __validate_hard_mode(self, guess: str) -> bool:
        last_guess, last_score = self.scores[-1]

        #Zip the guess and score together along with the position of the letter
        zip_list = [
            (letter, score, position)
            for position, (letter, score) 
            in enumerate(zip(last_guess, last_score))
        ]

        greens = [
            (letter, position)
            for letter, score, position
            in zip_list
            if score == 2
        ]

        yellows = [
            (letter, position)
            for letter, score, position
            in zip_list
            if score == 1
        ]

        guess_list = list(guess)

        #Check greens first
        for letter, position in greens:
            if guess[position] != letter:
                return False
            
            #Remove the letter from the guess list
            guess_list.pop(position)
        
        #Check yellows
        for letter, position in yellows:
            if letter not in guess_list:
                return False
            else:
                guess_list.remove(letter)
        
        return True