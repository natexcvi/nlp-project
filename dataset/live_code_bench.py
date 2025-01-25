from typing import Any

from testcontainers.core.container import DockerContainer

from dataset import Problem


class LiveCodeBenchLite:
    def __init__(self, score_utils):
        self.score_utils = score_utils
        self.__problems = [
            self.__create_instance(
                "sample_problem",
                "You are given a positive integer array 'nums'. Return the total frequencies of elements in 'nums' such that those elements all have the maximum frequency.",
                [
                    ([1, 3, 3, 4, 4], 4),
                ],
            )
        ]

    def __extract_solution(self, output):
        start_tag = "<solution>"
        end_tag = "</solution>"
        start_index = output.find(start_tag) + len(start_tag)
        end_index = output.find(end_tag)
        if start_index == -1 or end_index == -1:
            return None
        return output[start_index:end_index].strip()

    def __create_scorer_fn(self, code: str, test_cases: list[tuple[Any, Any]]):
        def scorer_fn(output):
            solution = self.__extract_solution(output)
            if not solution:
                print(f"No solution found in output: {output}")
                return 0
            with DockerContainer("python:3.9") as container:
                container.with_command("tail -f /dev/null").start()
                retcode, retval = container.exec(f"echo '{code}' > /code.py")
                if retcode != 0:
                    print(f"Failed to write code '{code}': {retval}")
                    return 0
                total_score = 0
                for test_case, expected_output in test_cases:
                    retcode, retval = container.exec(
                        f"echo '{test_case}' > /test_case.py"
                    )
                    if retcode != 0:
                        print(f"Failed to write test case '{test_case}': {retval}")
                        continue
                    retcode, retval = container.exec(
                        f"python /code.py '{test_case}' > /output.txt"
                    )
                    if retcode != 0:
                        print(f"Failed to run test case '{test_case}': {retval}")
                        continue
                    retcode, output = container.exec("cat /output.txt")
                    if retcode != 0:
                        print(f"Failed to run test case '{test_case}'")
                        continue
                    if output.decode("utf-8").strip() != expected_output:
                        print(
                            f"Test case '{test_case}' failed. Expected '{expected_output}', got '{output}'"
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
                          Format your solution as a Python code snippet wrapped in <solution> and </solution> tags.
                          The code should accept a single string argument and print the result to stdout.""",
            scorer_fn=self.__create_scorer_fn(statement, test_cases),
        )

    @property
    def problems(self):
        return self.__problems
