# agents_mechanism

A standalone agent similar to Onyx's deep research agent. It uses [LiteLLM](https://github.com/BerriAI/litellm), so any supported provider works, and it searches the web with DuckDuckGo (no API key needed for search).

## How it works

1. **Clarify** (optional): the LLM asks clarifying questions or moves on.
2. **Plan**: a numbered research plan of at most 6 steps.
3. **Research**: an orchestrator delegates tasks to parallel research sub-agents. Each sub-agent uses `web_search` and `open_url`, then writes an intermediate report with citations.
4. **Report**: the final report is written from all findings, with citations renumbered into one global source list.

## Usage

```bash
export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY, etc.
uv run scripts/deep_research_agent.py "What changed in Python 3.13?"
uv run scripts/deep_research_agent.py "..." --model anthropic/claude-sonnet-5-5 -o report.md
```

Useful flags: `--model` (or `$DR_MODEL`), `--skip-clarification`, `--reasoning-model` / `--non-reasoning-model`, `-o/--output`, `-q/--quiet`.

## Layout

| File | Purpose |
|---|---|
| `scripts/deep_research_agent.py` | Orchestration loop and CLI |
| `scripts/research_agent.py` | Research sub-agent |
| `scripts/prompts.py` | Prompt templates and tool names |
| `scripts/tools.py` | Tool definitions |
| `scripts/llm.py` | LiteLLM wrapper and message helpers |
| `scripts/web_tools.py` | Web search, page fetching, citations |
| `scripts/utils.py` | Console output and small helpers |

## Planning & Orchestrator
<img width="1900" height="1600" alt="image" src="https://github.com/user-attachments/assets/88f2e0bb-595f-4eed-a198-2a580c3f93a0" />


## Research Agent (Sub-Agent)
<img width="1900" height="1400" alt="image" src="https://github.com/user-attachments/assets/92b5d4f8-241b-4031-890f-c61125306247" />
