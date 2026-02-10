from __future__ import annotations

from typing import Any

from crewai.events.event_bus import crewai_event_bus
from crewai.llm import BaseLLM
from rlm import RLM

from experimental_rlms.events import RLMIterationEvent


class RLMEventLogger:
    """Duck-typed RLM logger that emits CrewAI events for each iteration."""

    def __init__(
        self,
        agent_role: str | None = None,
        agent_id: str | None = None,
    ):
        self._agent_role = agent_role
        self._agent_id = agent_id
        self._iteration_count = 0

    @property
    def iteration_count(self) -> int:
        return self._iteration_count

    def log_metadata(self, metadata: Any) -> None:
        """No-op — metadata logging not needed for event emission."""

    def log(self, iteration: Any) -> None:
        """Emit an RLMIterationEvent for this iteration."""
        self._iteration_count += 1

        code_snippets: list[str] = []
        code_outputs: list[str] = []
        code_errors: list[str] = []

        for block in iteration.code_blocks:
            code_snippets.append(block.code)
            code_outputs.append(block.result.stdout)
            code_errors.append(block.result.stderr)

        event = RLMIterationEvent(
            iteration_number=self._iteration_count,
            code_snippets=code_snippets,
            code_outputs=code_outputs,
            code_errors=code_errors,
            execution_time=iteration.iteration_time,
            has_final_answer=iteration.final_answer is not None,
            agent_role=self._agent_role,
            agent_id=self._agent_id,
        )
        crewai_event_bus.emit(self, event)


class RLMLLM(BaseLLM):
    """Custom CrewAI LLM that uses Recursive Language Models (RLM) under the hood."""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        backend: str = "openai",
        temperature: float | None = None,
        max_iterations: int = 30,
        environment: str = "local",
        environment_kwargs: dict[str, Any] | None = None,
        verbose: bool = False,
        context_data: Any | None = None,
        agent_role: str | None = None,
        agent_id: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, temperature=temperature, **kwargs)

        logger = None
        if agent_role or agent_id:
            logger = RLMEventLogger(agent_role=agent_role, agent_id=agent_id)

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
            logger=logger,
        )
        self._context_data = context_data

    def set_context_data(self, data: Any) -> None:
        """Set context data to be passed as the RLM prompt (for chunked processing).

        When set, the agent's messages become the root_prompt and this data
        becomes the prompt, enabling RLM's recursive decomposition over
        large inputs.
        """
        self._context_data = data

    @staticmethod
    def _format_context(data: Any) -> str:
        """Format context data as clean delimited text for the RLM prompt."""
        if isinstance(data, dict):
            sections = []
            for title, text in data.items():
                sections.append(f"{'=' * 60}\nDOCUMENT: {title}\n{'=' * 60}\n{text}")
            return "\n\n".join(sections)
        return str(data)

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
                prompt=self._format_context(self._context_data),
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
