from pydantic import BaseModel, Field


class RegexResponse(BaseModel):
    regex: str
    reasoning: str


class RegexGeneratedExamples(BaseModel):
    string_matches: list[str] = Field(
        description="List of 5 strings that should match the regex."
    )
    string_mismatches: list[str] = Field(
        description="List of 5 strings that should not match the regex."
    )
    reasoning: str
