import os

from pydantic import BaseModel

from nlp_project.dataset.algo_problems import AlgoProblems
from nlp_project.dataset.score_utils import ScoreUtils
from nlp_project.solvers.base_solver import Solver
from nlp_project.solvers.chain_of_thought import ChainOfThoughtSolver
from nlp_project.dataset.game_of_24 import GameOf24


class EvaluationResult(BaseModel):
    scores: list[float]

    @property
    def avg_score(self):
        return sum(self.scores) / len(self.scores)


NUM_ITERATIONS = 5

if __name__ == "__main__":
    solver: Solver = ChainOfThoughtSolver(
        "You are an algorithm expert."
    )
    score_utils = ScoreUtils()
    algo_problems = GameOf24(score_utils)

    for problem in algo_problems.problems:
        result = EvaluationResult(scores=[])
        for i in range(NUM_ITERATIONS):
            print(f"Evaluating problem '{problem.name}' ({i+1}/{NUM_ITERATIONS})...")
            output = solver.solve(problem)
            # print(f"Output: {output}")
            score = problem.scorer_fn(output)
            print(f"Score: {score}")
            result.scores.append(score)
        print(f"Average score for problem '{problem.name}': {result.avg_score:.1f}")
