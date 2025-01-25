from abc import ABC, abstractmethod

from openai import OpenAI

from dataset import Problem


class Solver(ABC):
    def __init__(self, openai_api_key: str):
        self.openai_client = OpenAI(api_key=openai_api_key)

    @abstractmethod
    def solve(self, problem: Problem) -> str:
        pass
