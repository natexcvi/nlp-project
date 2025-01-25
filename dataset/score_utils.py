import openai
import sympy as sp
from pydantic import BaseModel


class SemanticContainment(BaseModel):
    found: bool
    extracted_match: str


class ScoreUtils:
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)

    def simplify_math(self, expression):
        simplified_expression = sp.simplify(expression)
        return str(simplified_expression)

    def contains_semantically(self, target, text):
        response = self.openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
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

        embedding_response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002", input=[match.extracted_match, target]
        )

        match_embedding = embedding_response.data[0].embedding
        target_embedding = embedding_response.data[1].embedding

        similarity = self.__cosine_similarity(match_embedding, target_embedding)
        normalized_similarity = (similarity + 1) / 2
        return normalized_similarity

    def __cosine_similarity(self, vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_vec1 = sum(a * a for a in vec1) ** 0.5
        norm_vec2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm_vec1 * norm_vec2)
