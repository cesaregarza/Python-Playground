from typing import Optional
from .wordle import Wordle, GameOverException, NotInWordListException, HardModeException
from .solvers.solver_abc    import WordleSolverABC
import pandas as pd
import random
from tqdm import tqdm

class WordleEvaluator:

    def __init__(self, wordle_instance: 'Wordle') -> None:
        self.game                           = wordle_instance
        self.results:Optional[pd.DataFrame] = None
    
    def evaluate_solver(self, solver: 'WordleSolverABC', sample:Optional[int] = None) -> pd.DataFrame:
        
        #Obtain all solution words from the wordle instance, then shuffle them
        solution_words = self.game.solution_word_list
        random.shuffle(solution_words)

        #If a sample size is specified, then only take that many words from the solution list
        if sample is not None:
            solution_words = solution_words[:sample]

        #Initialize the recording list
        scores = []
        tqdm_solution = tqdm(solution_words)

        #Iterate through each solution word
        for solution_word in tqdm_solution:
            #Start the game
            self.game.start_game_select_word(solution_word)

            #Initialize the solver
            solver.initialize(self.game)

            #Initialize a guess, this will used for a while loop and will not actually be used
            guess       = ""
            num_guesses = 0
            while guess != solution_word:
                num_guesses += 1
                guess = solver.generate_guess()
                try:
                    results = self.game.submit_guess(guess)
                    solver.pass_results(results)
                except GameOverException:
                    break
                except NotInWordListException:
                    solver.not_in_word_list()
                    num_guesses -= 1
            
            #Add the score to the list
            scores += [{"solution_word": solution_word, "num_guesses": num_guesses}]
            mean_score = sum(score["num_guesses"] for score in scores) / len(scores)
            tqdm_solution.set_postfix({"mean_score": mean_score})
        
        #Convert the list to a dataframe
        self.results = pd.DataFrame(scores).set_index("solution_word")
        return self.results