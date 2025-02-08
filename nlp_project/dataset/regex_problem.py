from nlp_project.dataset.base_problem import Problem
from nlp_project.dataset.score_utils import ScoreUtils


class RegexProblems:
    def __init__(self, score_utils: ScoreUtils):
        regex_descriptions, regexes = self._read_regex_files()
        self.__problems = [
            Problem(
                name=regex_description,
                statement=regex_description,
                scorer_fn=lambda output: score_utils.compare_regexes(
                    regex, output
                ),
            ) for regex_description, regex in zip(regex_descriptions, regexes)
        ]

    @staticmethod
    def _read_regex_files() -> (list[str], list[str]):
        with open('data/KB13/regex_description.txt', 'r') as f:
            regex_description = f.readlines()
        with open('data/KB13/regexes.txt', 'r') as f:
            regexes = f.readlines()

        return regex_description, regexes

    @property
    def problems(self):
        return self.__problems
