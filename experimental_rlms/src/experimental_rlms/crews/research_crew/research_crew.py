from typing import Any, List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, before_kickoff, crew, task

from experimental_rlms.rlm_llm import RLMLLM


@CrewBase
class ResearchCrew:
    """Research Crew — analyzes papers using RLM's recursive decomposition."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # Set after construction, before .crew() is called
    papers_dict: dict[str, str] = {}

    @before_kickoff
    def _set_llm_context(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Push papers_dict into each RLMLLM agent right before execution."""
        for agent_obj in self.agents:
            if isinstance(agent_obj.llm, RLMLLM):
                agent_obj.llm.set_context_data(self.papers_dict)
        return inputs

    @agent
    def research_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["research_analyst"],  # type: ignore[index]
            llm=RLMLLM(
                model="gpt-4.1-nano",
                verbose=True,
                agent_role="research_analyst",
            ),
        )

    @task
    def literature_review(self) -> Task:
        return Task(
            config=self.tasks_config["literature_review"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
