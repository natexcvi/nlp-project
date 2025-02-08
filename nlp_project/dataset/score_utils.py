import re

import sympy as sp
from pydantic import BaseModel

from nlp_project.clients.openai_client import LLMConfig, get_openai_client


class SemanticContainment(BaseModel):
    found: bool
    extracted_match: str


class RegexResponse(BaseModel):
    cot_output: str
    regex: str
    original_instruction: str


class ScoreUtils:
    def __init__(self):
        self.openai_client = get_openai_client()
        self.llm_config = LLMConfig()

    def simplify_math(self, expression):
        simplified_expression = sp.simplify(expression)
        return str(simplified_expression)

    def evaluate_math(self, expression):
        return sp.sympify(expression).evalf()

    def extract_literals(self, expression):
        split_expression = sp.srepr(expression).split(" ")
        split_expression = [
            "".join([c for c in s if c.isnumeric()]) for s in split_expression
        ]
        return [s for s in split_expression if s]

    def contains_semantically(self, target, text):
        response = self.openai_client.beta.chat.completions.parse(
            model=self.llm_config.model,
            messages=[
                {"role": "system", "content": "You find matches in texts."},
                {
                    "role": "user",
                    "content": f"If the following text contains the term '{target}' or another term highly similar to it, return it. Here is the text:\n\n```\n{text}\n```",
                },
            ],
            response_format=SemanticContainment,
        )

        match = response.choices[0].message.parsed
        print(f"Match: {match}")
        if not match.found:
            return 0
        elif match.extracted_match not in text:
            raise ValueError("Extracted text does not appear in the original")

        embedding_response = self.openai_client.embeddings.create(
            model=self.llm_config.embeddings_model,
            input=[match.extracted_match, target],
        )

        match_embedding = embedding_response.data[0].embedding
        target_embedding = embedding_response.data[1].embedding

        similarity = self.__cosine_similarity(match_embedding, target_embedding)
        normalized_similarity = (similarity + 1) / 2
        return normalized_similarity

    @staticmethod
    def __cosine_similarity(vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_vec1 = sum(a * a for a in vec1) ** 0.5
        norm_vec2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm_vec1 * norm_vec2)

    def compare_regexes(self, regex_response: RegexResponse, regex2: str):
        regex1 = regex_response.regex
        return regex1 == regex2

    def is_regex_working_on_sample(self, regex_response: RegexResponse, matching_line: str):
        return re.match(regex_response.regex,  matching_line) is not None
