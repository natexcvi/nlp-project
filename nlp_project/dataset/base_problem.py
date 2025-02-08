from typing import Callable, Optional

from pydantic import BaseModel


class Problem(BaseModel):
    name: str
    statement: str
    scorer_fn: Callable[[str], float]
    response_format: Optional[type[BaseModel]] = None
