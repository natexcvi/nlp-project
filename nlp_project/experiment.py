import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import yaml
from pydantic import BaseModel, RootModel

from nlp_project.dataset.regex_problem import RegexProblems
from nlp_project.dataset.score_utils import ScoreUtils
from nlp_project.solvers.chain_of_thought import ChainOfThoughtSolver
from nlp_project.solvers.dyfs import DynamicFewShotSolver


class EvaluationResult(BaseModel):
    scores: list[float]
    outputs: list[Any]

    @property
    def avg_score(self):
        return sum(self.scores) / len(self.scores)


class IndividualResult(BaseModel):
    regex: str
    score: float


class TokenUsageStats(BaseModel):
    input_tokens: int
    output_tokens: int


class ProblemReport(BaseModel):
    avg_score: float
    results: list[IndividualResult]
    token_usage: TokenUsageStats


class SolverReport(RootModel):
    root: dict[str, ProblemReport]


class ExperimentReport(RootModel):
    root: dict[str, SolverReport]


class ExperimentSummary(BaseModel):
    total_problems: int
    total_solvers: int
    avg_score: float
    avg_score_per_model: dict[str, float]
    avg_tokens_per_model: dict[str, TokenUsageStats]
    num_iterations: int


NUM_ITERATIONS = 5
REPORT_FILE = "experiment_report.yaml"


def evaluate_problem(solver, problem, solver_name):
    result = EvaluationResult(scores=[], outputs=[])
    total_input_tokens = 0
    total_output_tokens = 0

    for i in range(NUM_ITERATIONS):
        print(
            f"Evaluating problem '{problem.name}' with {solver_name} ({i+1}/{NUM_ITERATIONS})..."
        )
        output = solver.solve(problem)
        score = problem.scorer_fn(output)
        print(f"Score: {score}, regex: {output.regex}")
        result.scores.append(score)
        result.outputs.append(output)

        total_input_tokens += solver.token_usage["input_tokens"]
        total_output_tokens += solver.token_usage["output_tokens"]

    avg_score = result.avg_score
    print(
        f"Average score for problem '{problem.name}' with {solver_name}: {avg_score:.1f}"
    )
    return (
        solver_name,
        problem.name,
        ProblemReport(
            results=[
                IndividualResult(regex=output.regex, score=score)
                for output, score in zip(result.outputs, result.scores)
            ],
            avg_score=avg_score,
            token_usage=TokenUsageStats(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            ),
        ),
    )


def generate_summary(report):
    unique_problems = set(
        problem_name
        for solver_report in report.values()
        for problem_name in solver_report
    )
    total_problems = len(unique_problems)
    total_solvers = len(report)
    avg_score = sum(
        problem_report.avg_score
        for solver_report in report.values()
        for problem_report in solver_report.values()
    ) / (total_problems * total_solvers)
    avg_score_per_model = {
        solver_name: sum(
            problem_report.avg_score for problem_report in solver_report.values()
        )
        / len(solver_report)
        for solver_name, solver_report in report.items()
    }
    avg_tokens_per_model = {
        solver_name: TokenUsageStats(
            input_tokens=sum(
                problem_report.token_usage.input_tokens
                for problem_report in solver_report.values()
            ),
            output_tokens=sum(
                problem_report.token_usage.output_tokens
                for problem_report in solver_report.values()
            ),
        )
        for solver_name, solver_report in report.items()
    }
    return ExperimentSummary(
        total_problems=total_problems,
        total_solvers=total_solvers,
        avg_score=avg_score,
        avg_score_per_model=avg_score_per_model,
        avg_tokens_per_model=avg_tokens_per_model,
        num_iterations=NUM_ITERATIONS,
    )


def run_experiment() -> None:
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
            for problem in algo_problems.problems[:20]:
                futures.append(
                    executor.submit(evaluate_problem, solver, problem, solver_name)
                )

        for future in as_completed(futures):
            solver_name, problem_name, problem_report = future.result()
            report[solver_name][problem_name] = problem_report

    experiment_report = ExperimentReport(root=report)
    summary = generate_summary(report)

    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(reports_dir, f"experiment_report_{timestamp}.yaml")

    with open(report_file, "w") as f:
        yaml.dump(
            {
                "summary": summary.model_dump(),
                "details": experiment_report.model_dump(),
            },
            f,
            default_flow_style=False,
        )
    print(f"Report saved to {report_file}")
