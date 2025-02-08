from typing import Callable

from pydantic import BaseModel


class Problem(BaseModel):
    name: str
    statement: str
    scorer_fn: Callable[[str], float]
