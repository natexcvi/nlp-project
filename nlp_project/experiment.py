import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from pydantic import BaseModel, RootModel

from nlp_project.dataset.algo_problems import AlgoProblems
from nlp_project.dataset.game_of_24 import GameOf24
from nlp_project.dataset.live_code_bench import LiveCodeBenchLite
from nlp_project.dataset.regex_problem import RegexProblems
from nlp_project.dataset.score_utils import ScoreUtils
from nlp_project.solvers.base_solver import Solver
from nlp_project.solvers.chain_of_thought import ChainOfThoughtSolver
from nlp_project.solvers.dyfs import DynamicFewShotSolver


class EvaluationResult(BaseModel):
    scores: list[float]

    @property
    def avg_score(self):
        return sum(self.scores) / len(self.scores)


class ProblemReport(BaseModel):
    scores: list[float]
    avg_score: float


class SolverReport(RootModel):
    root: dict[str, ProblemReport]


class ExperimentReport(RootModel):
    root: dict[str, SolverReport]


NUM_ITERATIONS = 5
REPORT_FILE = "experiment_report.yaml"


def evaluate_problem(solver, problem, solver_name):
    result = EvaluationResult(scores=[])
    for i in range(NUM_ITERATIONS):
        print(
            f"Evaluating problem '{problem.name}' with {solver_name} ({i+1}/{NUM_ITERATIONS})..."
        )
        output = solver.solve(problem)
        # print(f"Output: {output}")
        score = problem.scorer_fn(output)
        print(f"Score: {score}, regex: {output.regex}")
        result.scores.append(score)
    avg_score = result.avg_score
    print(
        f"Average score for problem '{problem.name}' with {solver_name}: {avg_score:.1f}"
    )
    return problem.name, ProblemReport(scores=result.scores, avg_score=avg_score)


def run_experiment():
    solvers = {
        "DynamicFewShotSolver": DynamicFewShotSolver(
            "Your task is to create a regex according to the user provided instructions."
        ),
        "ChainOfThoughtSolver": ChainOfThoughtSolver(
            "Your task is to create a regex according to the user provided instructions."
        ),
    }
    score_utils = ScoreUtils()
    algo_problems = RegexProblems(score_utils)
    report = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for solver_name, solver in solvers.items():
            report[solver_name] = {}
            for problem in algo_problems.problems:
                futures.append(
                    executor.submit(evaluate_problem, solver, problem, solver_name)
                )

        for future in as_completed(futures):
            solver_name, problem_report = future.result()
            report[solver_name][problem_report.name] = problem_report

    experiment_report = ExperimentReport(root=report)
    with open(REPORT_FILE, "w") as f:
        yaml.dump(experiment_report.model_dump(), f, default_flow_style=False)
    print(f"Report saved to {REPORT_FILE}")
