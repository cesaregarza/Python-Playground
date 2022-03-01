# %%
from wordle import Wordle, WordleEvaluator
# from wordle.naive_solver import NaiveSolver
from wordle.solvers.entropy import OneStepEntropicSolver, EntropicSolver
import requests

solution_url = r"https://gist.githubusercontent.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/5d752e5f0702da315298a6bb5a771586d6ff445c/wordle-answers-alphabetical.txt"
allowed_url  = r"https://gist.githubusercontent.com/cfreshman/cdcdf777450c5b5301e439061d29694c/raw/de1df631b45492e0974f7affe266ec36fed736eb/wordle-allowed-guesses.txt"
# %%
solution_words = requests.get(solution_url).text.split('\n')
allowed_words  = requests.get(allowed_url).text.split('\n')

all_words = list(set(solution_words + allowed_words))

# %%
wordle = Wordle(solution_word_list= solution_words, allowed_word_list= allowed_words, max_guesses=6)

if __name__ == "__main__":
    evaluator = WordleEvaluator(wordle)
    # entropic_solver = EntropicSolver(allowed_words, solution_words, r"F:\Dev\Python-Playground\wordle\precompute\word_matrix.csv")
    # entropic_solver = OneStepEntropicSolver(allowed_words, solution_words, r"F:\Dev\Python-Playground\wordle\precompute\word_matrix.csv")
    entropic_solver = OneStepEntropicSolver(all_words, precomputed_word_matrix_path=r"F:\Dev\Python-Playground\wordle\precompute\word_matrix_full.csv")
    # entropic_solver = OneStepEntropicSolver(all_words)
    evaluator.evaluate_solver(entropic_solver)
# %%
# %load_ext line_profiler
%lprun -f EntropicSolver._compute_entropy evaluator.evaluate_solver(entropic_solver, sample=50)
# %%
