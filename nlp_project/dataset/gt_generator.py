import json
import re

from pydantic import BaseModel, Field
from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError

from nlp_project.clients.openai_client import LLMConfig, get_openai_client
from nlp_project.dataset.regex_problem import WORKING_DIR


class StringSampleResponse(BaseModel):
    string_matches: list[str] = Field(description="The list of strings that match the regex pattern")
    string_mismatches: list[str] = Field(description="The list of strings that do not match the regex pattern")


class GTGenerator:
    def __init__(self):
        self.openai_client = get_openai_client()
        self.llm_config = LLMConfig()

    def validate_samples(self, samples: StringSampleResponse, regex: str):
        """
        Validate the generated string samples against the regex pattern.

        Args:
            samples: The generated string samples.
            regex: The regex pattern to validate the samples against.

        Returns:
            None
        """
        passed_samples = []
        for sample in samples.string_matches:
            if re.match(regex, sample):
                passed_samples.append(sample)
        samples.string_matches = passed_samples.copy()

        passed_samples = []
        for sample in samples.string_mismatches:
            if not re.match(regex, sample):
                passed_samples.append(sample)
        samples.string_mismatches = passed_samples.copy()

        if len(samples.string_matches) < 2 or len(samples.string_mismatches) < 2:
            raise ValueError("No valid samples found")

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def generate_regex_string_samples(self, regex_instructions: str, regex: str) -> StringSampleResponse:
        """
        Generate string samples that match a given regex pattern.

        Args:
            regex_instructions: The regex pattern to generate string samples for.

        Returns:
            A StringSampleResponse object containing the generated string samples.
        """

        response = self.openai_client.beta.chat.completions.parse(
            model=self.llm_config.model,
            messages=[
                {"role": "system", "content": "Generate string samples that match a given instruction."},
                {
                    "role": "user",
                    "content": f"Generate at least 3 string samples that match the following instructions and at least 3 that do not: {regex_instructions}",
                },
            ],
            response_format=StringSampleResponse,
        )

        samples = response.choices[0].message.parsed
        try:
            self.validate_samples(samples, regex)
        except ValueError as e:
            print(f"Error: {e} on regex pattern: {regex_instructions} and samples: {samples}")
            raise e

        return samples


if __name__ == "__main__":
    gt_generator = GTGenerator()

    data_file_path = WORKING_DIR.parent / 'data' / 'KB13'
    with open(data_file_path / 'regexes.txt', "r") as f:
        regexes = f.readlines()
    with open(data_file_path / 'regex_descriptions.txt', "r") as f:
        regexes_instructions = f.readlines()

    samples = {}
    for regex, instructions in list(zip(regexes, regexes_instructions))[:3]:
        instructions = instructions.strip()
        regex = regex.strip()
        try:
            res = gt_generator.generate_regex_string_samples(instructions, regex)
            samples[instructions] = {**res.model_dump(), "regex": regex}
        except RetryError as e:
            print(f"Error: {e} on regex pattern: {instructions}")
            continue

    with open(data_file_path / 'samples.json', 'w') as f:
        json.dump(samples, f, indent=4)
