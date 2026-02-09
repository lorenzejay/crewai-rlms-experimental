from typing import Any

from crewai.llm import BaseLLM
from rlm import RLM


class RLMLLM(BaseLLM):
    """Custom CrewAI LLM that uses Recursive Language Models (RLM) under the hood."""

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        backend: str = "openai",
        temperature: float | None = None,
        max_iterations: int = 30,
        environment: str = "local",
        environment_kwargs: dict[str, Any] | None = None,
        verbose: bool = False,
        context_data: Any | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, temperature=temperature, **kwargs)

        backend_kwargs = {"model_name": model}
        if temperature is not None:
            backend_kwargs["temperature"] = temperature

        self.rlm = RLM(
            backend=backend,
            backend_kwargs=backend_kwargs,
            environment=environment,
            environment_kwargs=environment_kwargs or {},
            max_iterations=max_iterations,
            verbose=verbose,
        )
        self._context_data = context_data

    def set_context_data(self, data: Any) -> None:
        """Set context data to be passed as the RLM prompt (for chunked processing).

        When set, the agent's messages become the root_prompt and this data
        becomes the prompt, enabling RLM's recursive decomposition over
        large inputs.
        """
        self._context_data = data

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        **kwargs,
    ) -> str:
        # Convert CrewAI message format to a single prompt string for RLM
        agent_prompt = self._messages_to_prompt(messages)

        if self._context_data is not None:
            # Pass context_data as prompt (for chunked processing) and
            # agent messages as root_prompt (visible to root LM every iteration)
            result = self.rlm.completion(
                prompt=str(self._context_data),
                root_prompt=agent_prompt,
            )
        else:
            result = self.rlm.completion(agent_prompt)
        return result.response

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    @staticmethod
    def _messages_to_prompt(messages) -> str:
        if isinstance(messages, str):
            return messages

        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multimodal content blocks - extract text parts
                text_parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)
