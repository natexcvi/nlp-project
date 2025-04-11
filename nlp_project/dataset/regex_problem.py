import json
import logging
import re

from nlp_project.clients.openai_client import WORKING_DIR
from nlp_project.dataset.base_problem import Problem
from nlp_project.dataset.gt_generator import RegexExamples
from nlp_project.dataset.regex_models import RegexGeneratedExamples, RegexResponse
from nlp_project.dataset.score_utils import ScoreUtils


class RegexProblems:
    def __init__(self, score_utils: ScoreUtils):
        self.regex_descriptions: list[str] = []
        self.regex_examlpes: list[RegexExamples] = []

        for regex_description, regex_example in self.__read_regex_examples().items():
            self.regex_descriptions.append(regex_description)
            self.regex_examlpes.append(regex_example)
        self.__problems = [
            Problem(
                name=regex_description,
                statement=regex_description,
                scorer_fn=lambda output, example=sample_string: score_utils.validate_against_test_cases(
                    output, example
                ),
                response_format=RegexResponse,
                solution_evaluator=lambda solution: lambda txt: self.safe_regex_match(
                    solution.regex, txt
                ),
            )
            for regex_description, sample_string in zip(
                self.regex_descriptions, self.regex_examlpes
            )
        ]

    def safe_regex_match(regex_str, text):
        try:
            return re.match(regex_str, text) is not None
        except Exception as e:
            logging.error(f"Error evaluating regex `{regex_str}` on text `{text}`: {e}")
            return False

    @staticmethod
    def __read_regex_examples() -> dict[str, RegexExamples]:
        data_dir = WORKING_DIR.parent / "data" / "KB13"
        with open(data_dir / "samples.json", "r") as f:
            regex_examples = json.load(f)
            regex_examples = {k: RegexExamples(**v) for k, v in regex_examples.items()}
        return regex_examples

    @property
    def problems(self):
        return self.__problems


class RegexExampleGenerationProblems(RegexProblems):
    def __init__(self, score_utils: ScoreUtils):
        super().__init__(score_utils)
        self.__problems = [
            Problem(
                name=regex_description,
                statement=regex_description,
                scorer_fn=lambda output, example=sample_strings: score_utils.validate_against_test_cases(
                    RegexResponse(
                        regex=example.regex,
                        reasoning="",
                    ),
                    RegexExamples(
                        regex=example.regex,
                        string_matches=output.string_matches,
                        string_mismatches=output.string_mismatches,
                    ),
                ),
                response_format=RegexGeneratedExamples,
                solution_evaluator=None,
            )
            for regex_description, sample_strings in zip(
                self.regex_descriptions, self.regex_examlpes
            )
        ]

    @property
    def problems(self):
        return self.__problems
