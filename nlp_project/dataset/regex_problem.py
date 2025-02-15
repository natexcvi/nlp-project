import json

from nlp_project.clients.openai_client import WORKING_DIR
from nlp_project.dataset.base_problem import Problem
from nlp_project.dataset.gt_generator import RegexExample
from nlp_project.dataset.score_utils import RegexResponse, ScoreUtils


class RegexProblems:
    def __init__(self, score_utils: ScoreUtils):
        regex_descriptions: list[str] = []
        regex_examlpes: list[RegexExample] = []
        for regex_description, regex_example in self.__read_regex_examples().items():
            regex_descriptions.append(regex_description)
            regex_examlpes.append(regex_example)
        self.__problems = [
            Problem(
                name=regex_description,
                statement=regex_description,
                scorer_fn=lambda output, example=sample_string: score_utils.validate_against_test_cases(
                    output, example
                ),
                response_format=RegexResponse,
            )
            for regex_description, sample_string in zip(
                regex_descriptions, regex_examlpes
            )
        ]

    @staticmethod
    def _read_regex_files() -> dict[str, str]:
        data_dir = WORKING_DIR.parent / "data" / "KB13"
        # with open(data_dir / 'regex_descriptions.txt', 'r') as f:
        #     regex_description = f.readlines()
        # with open(data_dir / 'regexes.txt', 'r') as f:
        #     regexes = f.readlines()
        with open(data_dir / "single_sample.json", "r") as f:
            regex_examples = json.load(f)

        return regex_examples

    @staticmethod
    def __read_regex_examples() -> dict[str, RegexExample]:
        data_dir = WORKING_DIR.parent / "data" / "KB13"
        with open(data_dir / "samples.json", "r") as f:
            regex_examples = json.load(f)
            regex_examples = {k: RegexExample(**v) for k, v in regex_examples.items()}
        return regex_examples

    @property
    def problems(self):
        return self.__problems
