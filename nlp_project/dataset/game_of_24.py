from nlp_project.dataset.base_problem import Problem


class GameOf24:
    def __init__(self, score_utils):
        self.score_utils = score_utils
        self.__problems = [
            self.__create_instance([1, 8, 12, 12]),
        ]

    def __extract_solution(self, output):
        start_tag = "<solution>"
        end_tag = "</solution>"
        start_index = output.find(start_tag) + len(start_tag)
        end_index = output.find(end_tag)
        if start_index == -1 or end_index == -1:
            return None
        return output[start_index:end_index].strip()

    def __create_scorer_fn(self, numbers: list[int]):
        def scorer_fn(output):
            solution = self.__extract_solution(output)
            if not solution:
                print(f"No solution found in output: {output}")
                return 0
            if self.score_utils.evaluate_math(solution) != 24:
                print(f"Solution '{solution}' does not evaluate to 24")
                return 0
            literals = self.score_utils.extract_literals(solution)
            if len(literals) != 4 or set(literals) != set([str(n) for n in numbers]):
                print(
                    f"Solution '{solution}' does not use all four numbers exactly once"
                )
                return 0
            return 1

        return scorer_fn

    def __create_instance(self, numbers: list[int]):
        return Problem(
            name=" ".join(str(n) for n in numbers),
            statement=f"""Use all four numbers exactly once to make 24.
                          You can use the four basic operations (+, -, *, /) and parentheses.
                          The numbers are {numbers}.
                          Format your solution as a mathematical expression
                          wrapped in <solution> and </solution> tags, without
                          the '=24' part. Do not use any special formatting.""",
            scorer_fn=self.__create_scorer_fn(numbers),
        )

    @property
    def problems(self):
        return self.__problems
