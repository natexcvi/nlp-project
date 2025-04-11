import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, RootModel
from tqdm import tqdm

from nlp_project.dataset.regex_problem import (
    RegexExampleGenerationProblems,
    RegexProblems,
)
from nlp_project.dataset.score_utils import ScoreUtils
from nlp_project.solvers.chain_of_thought import ChainOfThoughtSolver


class EvaluationResult(BaseModel):
    scores: list[float]
    outputs: list[Any]
    conversations: list[List[Dict[str, Any]]]

    @property
    def avg_score(self):
        return sum(self.scores) / len(self.scores)


class IndividualResult(BaseModel):
    output: Any
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


class ConversationReport(BaseModel):
    solver_name: str
    problem_name: str
    iteration: int
    conversation: List[Dict[str, Any]]


class ConversationsReport(RootModel):
    root: List[ConversationReport]


class ExperimentSummary(BaseModel):
    total_problems: int
    total_solvers: int
    avg_score: float
    avg_score_per_model: dict[str, float]
    total_tokens_per_model: dict[str, TokenUsageStats]
    num_iterations: int


NUM_ITERATIONS = 5
REPORT_FILE = "experiment_report.yaml"
CONVERSATIONS_FILE = "conversations_report.yaml"


def evaluate_problem(solver, problem, solver_name):
    result = EvaluationResult(scores=[], outputs=[], conversations=[])
    total_input_tokens = 0
    total_output_tokens = 0

    for i in tqdm(
        range(NUM_ITERATIONS), desc=f"{solver_name} - {problem.name}", leave=False
    ):
        output, conversation = solver.solve(problem)
        score = problem.scorer_fn(output)
        result.scores.append(score)
        result.outputs.append(output)
        result.conversations.append(conversation)

        total_input_tokens += solver.token_usage["input_tokens"]
        total_output_tokens += solver.token_usage["output_tokens"]

    avg_score = result.avg_score
    tqdm.write(
        f"Average score for problem '{problem.name}' with {solver_name}: {avg_score:.1f}"
    )

    conversation_reports = [
        ConversationReport(
            solver_name=solver_name,
            problem_name=problem.name,
            iteration=i + 1,
            conversation=conversation,
        )
        for i, conversation in enumerate(result.conversations)
    ]

    return (
        solver_name,
        problem.name,
        ProblemReport(
            results=[
                IndividualResult(
                    output=output,
                    score=score,
                )
                for output, score in zip(result.outputs, result.scores)
            ],
            avg_score=avg_score,
            token_usage=TokenUsageStats(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            ),
        ),
        conversation_reports,
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
    total_tokens_per_model = {
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
        total_tokens_per_model=total_tokens_per_model,
        num_iterations=NUM_ITERATIONS,
    )


def run_experiment(sample_size: Optional[int] = None) -> None:
    solvers = {
        # "DynamicFewShotSolver": DynamicFewShotSolver(
        #     "Your task is to create a regex according to the user provided instructions."
        # ),
        "ChainOfThoughtSolver-BuildRegex": ChainOfThoughtSolver(
            "Your task is to create a regex according to the user provided instructions."
        ),
        "ChainOfThoughtSolver-FindExamples": ChainOfThoughtSolver(
            "Your task is to find examples that match/don't match the regex described in the user provided instructions."
        ),
    }
    score_utils = ScoreUtils()
    regex_problem_set = RegexProblems(score_utils)
    regex_examples_problem_set = RegexExampleGenerationProblems(score_utils)

    assert len(regex_problem_set.problems) == len(
        regex_examples_problem_set.problems
    ), "Problem sets must be of equal size"

    problem_sample_index = random.sample(
        range(len(regex_problem_set.problems)), sample_size
    )

    solver_problem_mapping = {
        "ChainOfThoughtSolver-BuildRegex": regex_problem_set.problems,
        "ChainOfThoughtSolver-FindExamples": regex_examples_problem_set.problems,
    }

    report = {}
    all_conversations = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        total_tasks = len(solvers) * len(problem_sample_index)

        print(
            f"Running experiment with {len(solvers)} solvers on {total_tasks} problems"
        )

        for solver_name, solver in solvers.items():
            report[solver_name] = {}
            problems = solver_problem_mapping[solver_name]
            sampled_problems = (problems[i] for i in problem_sample_index)
            for problem in sampled_problems:
                futures.append(
                    executor.submit(evaluate_problem, solver, problem, solver_name)
                )

        for future in tqdm(
            as_completed(futures), total=total_tasks, desc="Overall progress"
        ):
            solver_name, problem_name, problem_report, conversation_reports = (
                future.result()
            )
            report[solver_name][problem_name] = problem_report
            all_conversations.extend(conversation_reports)

    experiment_report = ExperimentReport(root=report)
    conversations_report = ConversationsReport(root=all_conversations)
    summary = generate_summary(report)

    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(reports_dir, f"experiment_report_{timestamp}.yaml")
    conversations_file = os.path.join(
        reports_dir, f"conversations_report_{timestamp}.yaml"
    )

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

    with open(conversations_file, "w") as f:
        yaml.dump(
            conversations_report.model_dump(),
            f,
            default_flow_style=False,
        )
    print(f"Conversations saved to {conversations_file}")
