from crewai.events.base_events import BaseEvent


class RLMIterationEvent(BaseEvent):
    """Event emitted for each RLM iteration during completion."""

    type: str = "rlm_iteration"
    iteration_number: int
    code_snippets: list[str] = []
    code_outputs: list[str] = []
    code_errors: list[str] = []
    execution_time: float | None = None
    has_final_answer: bool = False
