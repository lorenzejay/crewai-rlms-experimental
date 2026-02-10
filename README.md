# Research Crew with RLMs

A CrewAI flow that fetches research papers (PDF or HTML), extracts their full text, and produces a structured literature review — using [Recursive Language Models (RLMs)](https://arxiv.org/abs/2512.24601) as the LLM backend instead of standard API calls.

## Why RLMs

Standard LLM calls pack everything — system prompt, agent backstory, task instructions, and your actual input — into a single context window. This works fine for short inputs, but quality degrades as context grows. The model doesn't fail; it just reasons worse. This is [context rot](https://research.trychroma.com/context-rot).

RLMs solve this by **never putting your large input into the context window at all**. Instead, the input is stored as a variable in a Python REPL environment. The model receives only small metadata about the input and writes code to interact with it — slicing, searching, chunking, and launching recursive sub-calls on pieces of it. The model decides its own decomposition strategy at runtime.

### When RLMs help

- **Multi-document synthesis** — analyzing several full-length papers, reports, or contracts
- **Dense aggregation** — tasks where the answer depends on processing most or all of the input
- **Long structured output** — detailed reports with cross-references across large source material
- **Codebase-wide analysis** — recursive decomposition mirrors code structure naturally


## How it works

The project has three layers:

```
CLI (urls + topic)
  → ResearchFlow        fetch URLs, extract text, save output
    → ResearchCrew      CrewAI agent + task definitions
      → RLMLLM          custom BaseLLM that wraps the RLM library
        → RLM            recursive decomposition over the papers
          → OpenAI API   gpt-4.1-nano as the base model
```

The key integration is the dual-prompt mode in `RLMLLM.call()`:

```python
# Large paper text → stored in REPL, recursively decomposed
# Agent task instructions → visible to the root LM every iteration
result = self.rlm.completion(
    prompt=str(papers_dict),         # large input
    root_prompt=agent_prompt,        # task instructions (always visible)
)
```

This keeps the agent's task context stable while RLM handles the heavy lifting of processing large documents.

## Installation

```bash
cd experimental_rlms
uv lock && uv sync
```

Requires Python 3.11–3.13. Set your `OPENAI_API_KEY` in a `.env` file at the project root.

## Usage

```bash
# With default papers (arxiv + chroma research)
uv run research

# Custom topic and URLs
uv run research --topic "context windows" \
  https://arxiv.org/pdf/2503.09572 \
  https://research.trychroma.com/context-rot

# Any mix of PDF and HTML sources
uv run research --topic "retrieval augmented generation" \
  https://arxiv.org/pdf/2005.11401 \
  https://arxiv.org/pdf/2312.10997
```

Output is written to `literature_review.md` in the project root.

## Project structure

```
src/experimental_rlms/
├── main.py                          # ResearchFlow — orchestrates fetch → analyze → save
├── rlm_llm.py                      # RLMLLM — CrewAI BaseLLM wrapper for RLM
└── crews/research_crew/
    ├── research_crew.py             # ResearchCrew — agent + task definitions
    └── config/
        ├── agents.yaml              # research_analyst role, goal, backstory
        └── tasks.yaml               # literature_review task spec (6-section output)
```

## References

- [Recursive Language Models](https://arxiv.org/abs/2512.24601) — Zhang, Kraska, Khattab (2025)
- [RLM blog post](https://alexzhang13.github.io/blog/2025/rlm/) — Alex Zhang
- [Context Rot](https://research.trychroma.com/context-rot) — Chroma Research
- [RLM library](https://github.com/alexzhang13/rlm) — GitHub
- [CrewAI docs](https://docs.crewai.com/) — Framework documentation
