from abc import ABC, abstractmethod

from openai import OpenAI

from dataset.algo_problems import AlgoProblem


class Solver(ABC):
    def __init__(self, openai_api_key: str):
        self.openai_client = OpenAI(api_key=openai_api_key)

    @abstractmethod
    def solve(self, problem: AlgoProblem) -> str:
        pass
