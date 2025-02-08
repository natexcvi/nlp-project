import os

from pydantic import BaseModel

from nlp_project.dataset.algo_problems import AlgoProblems
from nlp_project.dataset.regex_problem import RegexProblems
from nlp_project.dataset.game_of_24 import GameOf24
from nlp_project.dataset.live_code_bench import LiveCodeBenchLite
from nlp_project.dataset.score_utils import ScoreUtils
from nlp_project.solvers.base_solver import Solver
from nlp_project.solvers.chain_of_thought import ChainOfThoughtSolver


class EvaluationResult(BaseModel):
    scores: list[float]

    @property
    def avg_score(self):
        return sum(self.scores) / len(self.scores)


NUM_ITERATIONS = 5


def run_experiment():
    solver: Solver = ChainOfThoughtSolver("Your task is to create a regex according to the following instructions:")
    score_utils = ScoreUtils()
    algo_problems = LiveCodeBenchLite(score_utils)
    algo_problems = RegexProblems(score_utils)

    for problem in algo_problems.problems:
        result = EvaluationResult(scores=[])
        for i in range(NUM_ITERATIONS):
            print(f"Evaluating problem '{problem.name}' ({i+1}/{NUM_ITERATIONS})...")
            output = solver.solve(problem)
            # print(f"Output: {output}")
            score = problem.scorer_fn(output)
            print(f"Score: {score}, regex: {output.regex}")
            result.scores.append(score)
        print(f"Average score for problem '{problem.name}': {result.avg_score:.1f}")
