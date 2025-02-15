import json
from pathlib import Path

from pydantic import BaseModel

from nlp_project.dataset.base_problem import Problem
from nlp_project.dataset.score_utils import RegexResponse, ScoreUtils

WORKING_DIR = Path(__file__).parent.parent


class RegexProblems:
    def __init__(self, score_utils: ScoreUtils):
        regex_descriptions = []
        regexes = []
        for regex_description, regex in self._read_regex_files().items():
            regex_descriptions.append(regex_description)
            regexes.append(regex)
        self.__problems = [
            Problem(
                name=regex_description,
                statement=regex_description,
                scorer_fn=lambda output, sample_string=sample_string: score_utils.is_regex_working_on_sample(
                    output, sample_string
                ),
                response_format=RegexResponse,
            )
            for regex_description, sample_string in zip(regex_descriptions, regexes)
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

    @property
    def problems(self):
        return self.__problems
