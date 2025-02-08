import os
from abc import ABC, abstractmethod

from pydantic import BaseModel

from nlp_project.clients.openai_client import get_openai_client, LLMConfig
from nlp_project.dataset.base_problem import Problem


class Solver(ABC):
    def __init__(self):
        self.openai_client = get_openai_client()
        self.llm_config = LLMConfig()

    @abstractmethod
    def solve(self, problem: Problem) -> BaseModel:
        pass
