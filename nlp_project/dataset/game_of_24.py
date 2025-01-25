from nlp_project.dataset import Problem


class GameOf24:
    def __init__(self, score_utils):
        scorer_fn = lambda output: float(score_utils.simplify_math(output)) == 24
        self.__problems = [
            Problem(
                name="1 8 12 12",
                statement="""Use all four numbers exactly once to make 24.
                             You can use the four basic operations (+, -, *, /) and parentheses.
                             The numbers are [1, 8, 12, 12].""",
                scorer_fn=scorer_fn,
            ),
        ]

    @property
    def problems(self):
        return self.__problems
