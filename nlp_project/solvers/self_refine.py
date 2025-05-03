from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from nlp_project.dataset.base_problem import Problem
from nlp_project.solvers.base_solver import Solver


class Feedback(BaseModel):
    issues: List[str] = Field(
        ...,
        description="List of issues with the current solution that need to be addressed",
    )
    suggestions: List[str] = Field(
        ..., description="List of suggestions for improving the solution"
    )


class SelfRefineSolver(Solver):
    def __init__(self, system_message: str, max_iterations: int = 1):
        """
        Initialize the SelfRefineSolver.

        Args:
            system_message: The system message to use for LLM calls
            max_iterations: Maximum number of refine iterations to perform
        """
        super().__init__()
        self.system_message = system_message
        self.max_iterations = max_iterations

    def __generate_feedback(
        self, problem: Problem, current_solution: Any
    ) -> Tuple[Feedback, List[Dict[str, Any]]]:
        """
        Generate feedback on the current solution.

        Args:
            problem: The problem being solved
            current_solution: The current solution to evaluate

        Returns:
            Tuple containing feedback and the feedback conversation
        """
        feedback_messages = [
            {
                "role": "system",
                "content": (
                    "You are responsible for providing constructive feedback on "
                    "a solution to help refine and improve it."
                ),
            },
            {
                "role": "user",
                "content": f"Here is the problem statement:\n\n{problem.statement}",
            },
            {
                "role": "user",
                "content": f"And here is the current solution:\n\n{current_solution}",
            },
            {
                "role": "user",
                "content": (
                    "Please provide constructive feedback on this solution. "
                    "Identify specific issues and provide concrete suggestions for improvement. "
                    "Focus on correctness, completeness, and edge cases the solution might not handle well."
                ),
            },
        ]

        response = self.openai_client.beta.chat.completions.parse(
            model=self.llm_config.model,
            messages=feedback_messages,
            response_format=Feedback,
        )

        feedback_messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )

        return response.choices[0].message.parsed, feedback_messages

    def __stringify_feedback(self, feedback: Feedback) -> str:
        """
        Convert feedback into a string format.

        Args:
            feedback: The feedback to stringify

        Returns:
            A string representation of the feedback
        """
        issues_str = "\n".join([f"- {issue}" for issue in feedback.issues])
        suggestions_str = "\n".join(
            [f"- {suggestion}" for suggestion in feedback.suggestions]
        )

        return f"Issues identified:\n{issues_str}\n\nSuggestions for improvement:\n{suggestions_str}"

    def solve(self, problem: Problem) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Solve the given problem with iterative self-refinement.

        Args:
            problem: The problem to solve

        Returns:
            A tuple containing:
            - The final solution to the problem
            - The complete conversation history
        """
        if issubclass(problem.response_format, BaseModel):
            completion_model = self.openai_client.beta.chat.completions.parse
        else:
            completion_model = self.openai_client.chat.completions.create

        # Initial solution
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": problem.statement},
            {
                "role": "user",
                "content": "Solve the problem step-by-step, reasoning about each step.",
            },
        ]

        response = completion_model(
            model=self.llm_config.model,
            messages=messages,
            response_format=problem.response_format,
        )

        self.token_usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

        conversation_history = []
        conversation = messages.copy()
        conversation.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )

        current_solution = (
            response.choices[0].message.parsed
            if hasattr(response.choices[0].message, "parsed")
            else response.choices[0].message.content
        )

        # Refinement loop
        for iteration in range(self.max_iterations):
            # If no evaluator available or no issues found, we're done
            if not problem.solution_evaluator:
                conversation_history.extend(conversation)
                return current_solution, conversation_history

            # Get feedback for refinement
            feedback, feedback_conversation = self.__generate_feedback(
                problem, current_solution
            )

            # If no issues found, we're done
            if not feedback.issues:
                conversation_history.extend(conversation)
                conversation_history.extend(feedback_conversation)
                return current_solution, conversation_history

            # Add feedback to conversation
            feedback_str = self.__stringify_feedback(feedback)
            conversation.append(
                {
                    "role": "user",
                    "content": f"Your solution needs refinement. Here's feedback to address:\n\n{feedback_str}\n\nPlease provide an improved solution that addresses these issues.",
                }
            )

            # Get refined solution
            response = completion_model(
                model=self.llm_config.model,
                messages=conversation,
                response_format=problem.response_format,
            )

            self.token_usage["input_tokens"] += response.usage.prompt_tokens
            self.token_usage["output_tokens"] += response.usage.completion_tokens

            conversation.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )

            current_solution = (
                response.choices[0].message.parsed
                if hasattr(response.choices[0].message, "parsed")
                else response.choices[0].message.content
            )

        # Add all conversations to history
        conversation_history.extend(conversation)

        return current_solution, conversation_history
