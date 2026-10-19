#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm>=1.60",
#     "ddgs>=9.0",
#     "httpx>=0.27",
#     "beautifulsoup4>=4.12",
# ]
# ///
"""Standalone deep research agent that follows the logic of Onyx's
`run_deep_research_llm_loop` (onyx/deep_research/dr_loop.py) without any Onyx
services (no Postgres, Redis, OpenSearch, or emitter/packet plumbing).

Flow:
  1. Clarification (optional): the LLM either asks the user clarifying questions
     or calls `generate_plan`.
  2. Research plan: a numbered list of <= 6 steps. The plan is only held in
     memory and re-injected into the orchestrator system prompt every cycle.
  3. Orchestrator loop: tool-only LLM calls choosing between `research_agent`,
     `think_tool` (non-reasoning models only) and `generate_report`.
     `research_agent` calls run in parallel as sub-agents that use `web_search`
     and `open_url`, then write an intermediate report with [n] citations.
  4. Final report: generated from the full research history. Sub-agent
     citations are renumbered into one global source list.

Internal (Onyx index) search is not available here; sub-agents only use the web.

Usage:
  export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY, etc. (any LiteLLM provider)
  uv run backend/scripts/deep_research_agent.py "What changed in Python 3.13?"
  uv run backend/scripts/deep_research_agent.py "..." --model anthropic/claude-sonnet-5-5 -o report.md
"""

import argparse
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import litellm

from llm import (
    LLMClient,
    Message,
    ToolCall,
    assistant_tool_call_message,
    find_tool_call,
    last_n_user_messages,
    system_message,
    tool_response_message,
    user_message,
)
from prompts import (
    CLARIFICATION_PROMPT,
    FINAL_REPORT_PROMPT,
    FIRST_CYCLE_REMINDER,
    GENERATE_REPORT_TOOL_NAME,
    ORCHESTRATOR_PROMPT,
    ORCHESTRATOR_PROMPT_REASONING,
    RESEARCH_AGENT_TOOL_NAME,
    RESEARCH_PLAN_PROMPT,
    RESEARCH_PLAN_REMINDER,
    THINK_TOOL_NAME,
    THINK_TOOL_RESPONSE_MESSAGE,
    USER_FINAL_REPORT_QUERY,
)
from research_agent import (
    RESEARCH_AGENT_FAILURE_MESSAGE,
    run_research_agents,
)
from tools import (
    GENERATE_PLAN_TOOL,
    GENERATE_REPORT_TOOL,
    RESEARCH_AGENT_TOOL,
    THINK_TOOL,
    ToolDefinition,
)
from utils import Console, get_current_datetime, shorten
from web_tools import (
    CitationMap,
    CitationRegistry,
    DuckDuckGoSearch,
    WebSearchProvider,
)

DEFAULT_MODEL = os.environ.get("DR_MODEL", "openai/gpt-5-mini")

# Orchestrator limits (dr_loop.py)
MAX_USER_MESSAGES_FOR_CONTEXT = 5
MAX_FINAL_REPORT_TOKENS = 20000
DEEP_RESEARCH_FORCE_REPORT_SECONDS = 30 * 60
MAX_ORCHESTRATOR_CYCLES = 8
MAX_ORCHESTRATOR_CYCLES_REASONING = 4
MIN_INPUT_TOKENS = 50000


@dataclass
class ClarificationRequest:
    questions: str


@dataclass
class ResearchReport:
    plan: str
    report: str
    sources: CitationMap


def run_deep_research(
    chat_history: list[Message],
    llm: LLMClient,
    search_provider: WebSearchProvider,
    console: Console,
    skip_clarification: bool = False,
    on_report_delta: Callable[[str], None] | None = None,
) -> ClarificationRequest | ResearchReport:
    """Runs one deep research turn: the user's latest message in, one reply out.

    1. Clarify  - optionally ask the user questions instead (ends the turn).
    2. Plan     - write a numbered research plan.
    3. Research - orchestrator loop delegating tasks to parallel research sub-agents.
    4. Report   - write the final answer from everything the sub-agents found.

    Only conversation text outlives the turn: the plan, the sub-agents' findings and
    the citation numbering are all local to this call.
    """
    _check_context_window(llm)
    start_time = time.monotonic()

    # Working copy of the conversation. The orchestrator appends its tool calls and the
    # sub-agents' reports to it, and the final report is written from it.
    history = list(chat_history)

    # 1. Clarify
    if not skip_clarification:
        questions = ask_clarifying_questions(history, llm, console)
        if questions is not None:
            return ClarificationRequest(questions=questions)

    # 2. Plan
    research_plan = generate_research_plan(history, llm, console)

    # 3. Research. Each sub-agent numbers its sources from [1]; the registry maps them
    # to one global numbering so the final report's citations are unambiguous.
    registry = CitationRegistry()
    run_orchestrator(
        history, research_plan, llm, search_provider, registry, console, start_time
    )

    # 4. Report
    report = generate_final_report(
        history, research_plan, llm, console, on_report_delta
    )
    return ResearchReport(
        plan=research_plan, report=report, sources=registry.cited_in(report)
    )


def _check_context_window(llm: LLMClient) -> None:
    # The research history holds several agent reports of up to ~10k tokens each.
    if llm.max_input_tokens is not None and llm.max_input_tokens < MIN_INPUT_TOKENS:
        raise RuntimeError(
            f"Cannot run Deep Research with an LLM that has less than {MIN_INPUT_TOKENS:,} max input tokens"
        )


def ask_clarifying_questions(
    history: list[Message], llm: LLMClient, console: Console
) -> str | None:
    """Step 1: returns questions for the user, or None if the query is clear enough.

    The model answers either with text (the questions) or by calling the generate_plan
    signal tool. That tool has no implementation; calling it just means "go ahead".
    """
    console.log("clarify", "checking whether clarification is needed")
    messages = [
        system_message(
            CLARIFICATION_PROMPT.format(current_datetime=get_current_datetime())
        ),
        *last_n_user_messages(history, MAX_USER_MESSAGES_FOR_CONTEXT),
    ]
    step = llm.tool_step(messages, [GENERATE_PLAN_TOOL], "auto")
    if step.tool_calls:
        return None
    return (step.content or "").strip()


def generate_research_plan(
    history: list[Message], llm: LLMClient, console: Console
) -> str:
    """Step 2: returns a numbered plan of at most 6 research steps.

    The plan is only held in memory. The orchestrator follows it because it is
    re-injected into the orchestrator's system prompt on every cycle.
    """
    console.log("plan", "generating research plan\n")
    messages = [
        system_message(
            RESEARCH_PLAN_PROMPT.format(current_datetime=get_current_datetime())
        ),
        # The trailing reminder stops models from answering the query instead of planning.
        *last_n_user_messages(
            history + [user_message(RESEARCH_PLAN_REMINDER)],
            MAX_USER_MESSAGES_FOR_CONTEXT + 1,
        ),
    ]
    plan = llm.stream_text(messages, max_tokens=None, on_delta=console.stream).strip()
    console.stream("\n\n")
    if not plan:
        raise RuntimeError("Deep Research failed to generate a research plan")
    return plan


def run_orchestrator(
    history: list[Message],
    research_plan: str,
    llm: LLMClient,
    search_provider: WebSearchProvider,
    registry: CitationRegistry,
    console: Console,
    start_time: float,
) -> None:
    """Step 3: asks the orchestrator LLM for its next action, cycle by cycle, until done.

    Each cycle is one LLM call that must respond with a tool call:
      - research_agent:  run the tasks as parallel sub-agents, add their reports to history
      - think_tool:      record the model's reasoning in history (non-reasoning models only)
      - generate_report: research is complete
    Everything is appended to `history`, which later cycles and the final report read.
    """
    # Reasoning models already think before every response, so they get no think_tool
    # and fewer cycles (no cycles spent on thinking).
    if llm.is_reasoning_model:
        max_cycles = MAX_ORCHESTRATOR_CYCLES_REASONING
        prompt_template = ORCHESTRATOR_PROMPT_REASONING
        tools = [RESEARCH_AGENT_TOOL, GENERATE_REPORT_TOOL]
    else:
        max_cycles = MAX_ORCHESTRATOR_CYCLES
        prompt_template = ORCHESTRATOR_PROMPT
        tools = [RESEARCH_AGENT_TOOL, GENERATE_REPORT_TOOL, THINK_TOOL]

    agents_started = 0
    for cycle in range(max_cycles):
        # Hard limits end research without consulting the LLM. The last cycle is
        # reserved for this, so the orchestrator makes at most max_cycles - 1 decisions.
        if time.monotonic() - start_time > DEEP_RESEARCH_FORCE_REPORT_SECONDS:
            console.log(
                "orchestrator",
                f"exceeded {DEEP_RESEARCH_FORCE_REPORT_SECONDS}s, forcing final report",
            )
            return
        if cycle == max_cycles - 1:
            console.log("orchestrator", "last cycle reached, forcing final report")
            return

        tool_calls = _ask_orchestrator(
            history, research_plan, prompt_template, tools, cycle, max_cycles, llm
        )

        # Act on the decision. Order matters when several tools are called at once:
        # finishing beats thinking, and thinking beats researching.
        if not tool_calls:
            if cycle == 0:
                raise RuntimeError(
                    "Deep Research failed to generate any research tasks for the agents."
                )
            console.log("orchestrator", "no tool calls returned, generating report")
            return

        if find_tool_call(tool_calls, GENERATE_REPORT_TOOL_NAME):
            console.log("orchestrator", f"cycle {cycle}: generate_report")
            return

        think_call = find_tool_call(tool_calls, THINK_TOOL_NAME)
        if think_call:
            _record_think(history, think_call, console, cycle)
            continue

        research_calls = [
            tc for tc in tool_calls if tc.name == RESEARCH_AGENT_TOOL_NAME
        ]
        if not research_calls:
            console.log(
                "orchestrator", "no research agent calls returned, generating report"
            )
            return

        _run_research_round(
            history,
            research_calls,
            llm,
            search_provider,
            registry,
            console,
            agents_started,
            cycle,
        )
        agents_started += len(research_calls)


def _ask_orchestrator(
    history: list[Message],
    research_plan: str,
    prompt_template: str,
    tools: list[ToolDefinition],
    cycle: int,
    max_cycles: int,
    llm: LLMClient,
) -> list[ToolCall]:
    """One orchestrator decision. tool_choice="required" means it can only act via tools."""
    # The plan exists nowhere else, so rebuilding the system prompt with it every cycle
    # is what keeps the orchestrator following it.
    system_prompt = prompt_template.format(
        current_datetime=get_current_datetime(),
        current_cycle_count=cycle,
        max_cycles=max_cycles,
        research_plan=research_plan,
    )
    messages = [
        system_message(system_prompt),
        *last_n_user_messages(history, MAX_USER_MESSAGES_FOR_CONTEXT),
    ]
    # Right after the first research round, push back against stopping too early.
    if cycle == 1:
        messages.append(user_message(FIRST_CYCLE_REMINDER))
    return llm.tool_step(messages, tools, "required").tool_calls


def _record_think(
    history: list[Message], think_call: ToolCall, console: Console, cycle: int
) -> None:
    """Stores the model's reasoning (the think_tool argument) so later cycles can read it.

    Nothing is executed. Any other tool calls made alongside it are dropped.
    """
    reasoning = str(think_call.args.get("reasoning", ""))
    console.log("orchestrator", f"cycle {cycle}: think: {shorten(reasoning, 400)}")
    history.append(assistant_tool_call_message([think_call]))
    # Every tool call needs a matching tool response; the content is just a placeholder.
    history.append(tool_response_message(think_call.id, THINK_TOOL_RESPONSE_MESSAGE))


def _run_research_round(
    history: list[Message],
    research_calls: list[ToolCall],
    llm: LLMClient,
    search_provider: WebSearchProvider,
    registry: CitationRegistry,
    console: Console,
    agents_started: int,
    cycle: int,
) -> None:
    """Runs the research_agent calls as parallel sub-agents and records their reports.

    The orchestrator only sees each sub-agent's final report, never its searches or
    fetched pages, which keeps the orchestrator's context small.
    """
    console.log(
        "orchestrator",
        f"cycle {cycle}: launching {len(research_calls)} research agent(s)",
    )
    reports = run_research_agents(
        research_calls, llm, search_provider, registry, console, agents_started
    )

    history.append(assistant_tool_call_message(research_calls))
    for call, report in zip(research_calls, reports):
        # Strict providers reject the next request if any tool call lacks a response,
        # so failed agents still get one.
        history.append(
            tool_response_message(call.id, report or RESEARCH_AGENT_FAILURE_MESSAGE)
        )


def generate_final_report(
    history: list[Message],
    research_plan: str,
    llm: LLMClient,
    console: Console,
    on_delta: Callable[[str], None] | None,
) -> str:
    """Step 4: writes the user-facing answer from the full research history.

    The plan is passed again as a structural guide, not a constraint.
    """
    console.log("report", "writing final report\n")
    messages = [
        system_message(
            FINAL_REPORT_PROMPT.format(current_datetime=get_current_datetime())
        ),
        *history,
        user_message(USER_FINAL_REPORT_QUERY.format(research_plan=research_plan)),
    ]
    report = llm.stream_text(messages, MAX_FINAL_REPORT_TOKENS, on_delta=on_delta)
    if not report:
        raise ValueError("LLM failed to generate the final deep research report")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def format_sources(sources: CitationMap) -> str:
    if not sources:
        return ""
    lines = [
        f"[{number}] {source.title} - {source.url}"
        for number, source in sources.items()
    ]
    return "\n\n## Sources\n" + "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Onyx-style deep research agent."
    )
    parser.add_argument(
        "query", nargs="?", help="Research question. Prompted for if omitted."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LiteLLM model string (default: {DEFAULT_MODEL}, or $DR_MODEL).",
    )
    parser.add_argument(
        "--skip-clarification", action="store_true", help="Go straight to planning."
    )
    reasoning = parser.add_mutually_exclusive_group()
    reasoning.add_argument(
        "--reasoning-model",
        dest="is_reasoning_model",
        action="store_true",
        default=None,
        help="Treat the model as a reasoning model (no think tool, fewer cycles).",
    )
    reasoning.add_argument(
        "--non-reasoning-model",
        dest="is_reasoning_model",
        action="store_false",
        help="Treat the model as a non-reasoning model (adds think tool).",
    )
    parser.set_defaults(is_reasoning_model=None)
    parser.add_argument(
        "-o", "--output", help="Also write the final report (markdown) to this file."
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Hide progress output on stderr."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    litellm.drop_params = True
    # Adds placeholder tools for providers (e.g. Anthropic) that reject tool-call history without tools.
    litellm.modify_params = True
    litellm.suppress_debug_info = True

    query = args.query or input("Research question: ").strip()
    if not query:
        print("A research question is required.", file=sys.stderr)
        return 2

    console = Console(quiet=args.quiet)
    llm = LLMClient(args.model, args.is_reasoning_model)
    console.log(
        "setup",
        f"model={llm.model} reasoning_model={llm.is_reasoning_model} max_input_tokens={llm.max_input_tokens or 'unknown'}",
    )

    def print_report_delta(delta: str) -> None:
        print(delta, end="", flush=True)

    chat_history: list[Message] = [user_message(query)]
    skip_clarification = args.skip_clarification
    while True:
        outcome = run_deep_research(
            chat_history,
            llm,
            DuckDuckGoSearch(),
            console,
            skip_clarification=skip_clarification,
            on_report_delta=print_report_delta,
        )
        if isinstance(outcome, ResearchReport):
            break

        print(outcome.questions)
        if not sys.stdin.isatty():
            print(
                "\n(Non-interactive input: re-run with your answers in the query, or pass --skip-clarification.)",
                file=sys.stderr,
            )
            return 1
        answer = input("\n> ").strip()
        # Mirrors Onyx: the turn after a clarification skips the clarification step.
        chat_history += [
            {"role": "assistant", "content": outcome.questions},
            user_message(answer),
        ]
        skip_clarification = True

    sources_text = format_sources(outcome.sources)
    print(sources_text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(outcome.report + "\n" + sources_text)
        console.log("done", f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
