import json
import re
from base64 import b64encode
from typing import Any

from testcontainers.core.container import DockerContainer

from nlp_project.dataset.base_problem import Problem


class LiveCodeBenchLite:
    def __init__(self, score_utils):
        self.score_utils = score_utils
        self.__problems = [
            self.__create_instance(
                "sample_problem",
                "You are given a positive integer array 'nums'. Return the total frequencies of elements in 'nums' such that those elements all have the maximum frequency.",
                [
                    (([1, 3, 3, 4, 4],), 4),
                    (([1, 2, 3, 4, 5],), 5),
                    (([1, 1, 2, 2, 3, 3],), 6),
                    (([8],), 1),
                    (([],), 0),
                ],
            )
        ]

    def __extract_solution(self, output):
        code_match = re.search(r"```python(.*?)```", output, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return None

    def __extract_function_name(self, code):
        match = re.search(r"def\s+(\w+)\s*\(.*?\):", code)
        if match:
            return match.group(1)
        return None

    def _prepare_solution(self, solution, *args):
        function_name = self.__extract_function_name(solution)
        if not function_name:
            print(f"Failed to extract function name from solution: {solution}")
            return None
        return f"{solution}\n\nimport json\nprint({function_name}(*{json.dumps(args)}))"

    def __create_scorer_fn(self, test_cases: list[tuple[tuple[Any], Any]]):
        def scorer_fn(output):
            solution = self.__extract_solution(output)
            if not solution:
                print(f"No solution found in output: {output}")
                return 0
            with DockerContainer("python:3.9") as container:
                container.with_command("tail -f /dev/null").start()
                total_score = 0
                for test_case, expected_output in test_cases:
                    code = self._prepare_solution(solution, *test_case)
                    retcode, retval = container.exec(
                        f'bash -c {json.dumps(f"echo {b64encode(code.encode()).decode()} | base64 -d > /code.py")}'
                    )
                    if retcode != 0:
                        print(f"Failed to write code: {retval}")
                        return 0
                    retcode, retval = container.exec(
                        f"bash -c 'python /code.py {json.dumps(test_case)}'"
                    )
                    if retcode != 0:
                        print(f"Failed to run test case '{test_case}': {retval}")
                        continue
                    if retval.decode("utf-8").strip() != str(expected_output):
                        print(
                            f"Test case '{test_case}' failed. Expected '{expected_output}', got '{retval.decode('utf-8').strip()}'"
                        )
                        continue
                    total_score += 1
                return total_score / len(test_cases)

        return scorer_fn

    def __create_instance(
        self, name: str, statement: str, test_cases: list[tuple[Any, Any]]
    ):
        return Problem(
            name=name,
            statement=f"""{statement}
                          Format your solution as a Python code snippet wrapped in ```python...```.
                          Your code should define a single function that takes the input as arguments
                          and returns the output.""",
            scorer_fn=self.__create_scorer_fn(test_cases),
        )

    @property
    def problems(self):
        return self.__problems
