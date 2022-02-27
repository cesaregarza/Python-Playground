from abc import ABC, abstractmethod
from ..wordle import Wordle

class WordleSolverABC(ABC):

    @abstractmethod
    def initialize(self, wordle_instance: 'Wordle') -> None:
        pass
    
    @abstractmethod
    def generate_guess(self):
        pass

    @abstractmethod
    def pass_results(self, results_list:list[int]) -> None:
        pass
    
    def not_in_word_list(self) -> None:
        pass