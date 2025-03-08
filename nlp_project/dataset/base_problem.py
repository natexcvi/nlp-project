from typing import Callable, Generic, Optional, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Problem(BaseModel, Generic[T]):
    name: str
    statement: str
    scorer_fn: Callable[[str], float]
    response_format: Optional[type[T]] = str
    solution_evaluator: Optional[Callable[[T], Callable]] = None
