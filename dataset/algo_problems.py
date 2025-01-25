from typing import Callable

from pydantic import BaseModel

from dataset.score_utils import ScoreUtils


class AlgoProblem(BaseModel):
    name: str
    statement: str
    scorer_fn: Callable[[str], float]


class AlgoProblems:
    def __init__(self, score_utils: ScoreUtils):
        self.__problems = [
            AlgoProblem(
                name="dfs run with maximum tree edges",
                statement="""Let (G = (V, E)) be a directed graph. As we have seen in class,
                             the DFS algorithm can have more than one possible output, due to the
                             degrees of freedom in both the representation and the algorithm itself.
                             Specifically, the classification of the edges (tree, back, forward, cross)
                             may vary depending on the specific execution.
                             Describe an algorithm as efficient as possible that finds a DFS run
                             on G in which the maximum number of edges are classified as tree edges.""",
                scorer_fn=lambda output: score_utils.contains_semantically(
                    "start from the node with the highest out degree", output
                ),
            ),
        ]

    @property
    def problems(self):
        return self.__problems
