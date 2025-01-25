import os

from openai import OpenAI
from pydantic import BaseModel

from dataset.algo_problems import AlgoProblems
from dataset.score_utils import ScoreUtils


class EvaluationResult(BaseModel):
    scores: list[float]

    @property
    def avg_score(self):
        return sum(self.scores) / len(self.scores)


NUM_ITERATIONS = 5

if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key)
    score_utils = ScoreUtils(openai_api_key)
    algo_problems = AlgoProblems(score_utils)

    for problem in algo_problems.problems:
        result = EvaluationResult(scores=[])
        for i in range(NUM_ITERATIONS):
            print(f"Evaluating problem '{problem.name}' ({i+1}/{NUM_ITERATIONS})...")
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an algorithms expert."},
                    {"role": "user", "content": problem.statement},
                ],
            )
            output = response.choices[0].message.content
            # print(f"Output: {output}")
            score = problem.scorer_fn(output)
            print(f"Score: {score}")
            result.scores.append(score)
        print(f"Average score for problem '{problem.name}': {result.avg_score:.1f}")
