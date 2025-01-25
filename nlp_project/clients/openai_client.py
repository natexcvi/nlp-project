import os

from openai import OpenAI
from pydantic import BaseModel


def get_openai_client() -> OpenAI:
    return OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/",
        api_key=os.getenv("GOOGLE_API_KEY"),
    )


class LLMConfig(BaseModel):
    model: str = "gemini-1.5-flash"
    embeddings_model: str = "models/text-embedding-004"  # "text-embedding-ada-002"