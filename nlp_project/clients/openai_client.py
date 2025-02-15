import os
from pathlib import Path

import toml
from openai import OpenAI
from pydantic import BaseModel

WORKING_DIR = Path(__file__).parent.parent


def get_openai_client(config: "LLMConfig") -> OpenAI:
    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )


class LLMConfig(BaseModel):
    model: str
    embeddings_model: str
    base_url: str
    api_key: str

    @classmethod
    def from_config_toml(cls) -> "LLMConfig":
        config = toml.load(WORKING_DIR / "clients/llm_config.toml")
        return cls(
            model=config.get("model", "gpt-4o"),
            embeddings_model=config.get("embeddings_model", "text-embedding-ada-002"),
            base_url=config.get("base_url", "https://api.openai.com/v1"),
            api_key=config["api_key"],
        )
