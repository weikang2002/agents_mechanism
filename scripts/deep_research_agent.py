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
import json
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import httpx
import litellm
from bs4 import BeautifulSoup

Message = dict[str, Any]
ToolDefinition = dict[str, Any]

DEFAULT_MODEL = os.environ.get("DR_MODEL", "openai/gpt-5-mini")

# Orchestrator limits (dr_loop.py)
MAX_USER_MESSAGES_FOR_CONTEXT = 5
MAX_FINAL_REPORT_TOKENS = 20000
DEEP_RESEARCH_FORCE_REPORT_SECONDS = 30 * 60
MAX_ORCHESTRATOR_CYCLES = 8
MAX_ORCHESTRATOR_CYCLES_REASONING = 4
MIN_INPUT_TOKENS = 50000

# Research sub-agent limits (tools/fake_tools/research_agent.py)
MAX_RESEARCH_CYCLES = 8
RESEARCH_AGENT_TIMEOUT_SECONDS = 30 * 60
RESEARCH_AGENT_TIMEOUT_MESSAGE = "Research Agent timed out after 30 minutes"
RESEARCH_AGENT_FORCE_REPORT_SECONDS = 12 * 60
MAX_INTERMEDIATE_REPORT_LENGTH_TOKENS = 10000
RESEARCH_AGENT_FAILURE_MESSAGE = "Research agent call failed. Try a different approach or continue without this result."

# Tool-call-only steps never need long outputs; the cap stops degenerate token loops.
# Only applied to non-reasoning models since reasoning tokens count against max_tokens.
TOOL_STEP_MAX_TOKENS = 1024

LLM_STEP_TIMEOUT_S = 300
LLM_REPORT_TIMEOUT_S = 900

# Web tools
MAX_SEARCH_QUERIES_PER_CALL = 3
SEARCH_RESULTS_PER_QUERY = 8
MAX_URLS_PER_CALL = 5
MAX_PAGE_CHARS = 20000
URL_FETCH_TIMEOUT_S = 20

GENERATE_PLAN_TOOL_NAME = "generate_plan"
RESEARCH_AGENT_TOOL_NAME = "research_agent"
RESEARCH_AGENT_TASK_KEY = "task"
GENERATE_REPORT_TOOL_NAME = "generate_report"
THINK_TOOL_NAME = "think_tool"
WEB_SEARCH_TOOL_NAME = "web_search"
OPEN_URL_TOOL_NAME = "open_url"
THINK_TOOL_RESPONSE_MESSAGE = "Acknowledged, please continue."


# ruff: noqa: E501 start
CLARIFICATION_PROMPT = f"""
You are a clarification agent that runs prior to deep research. Assess whether you need to ask clarifying questions, or if the user has already provided enough information for you to start research. \
CRITICAL - Never directly answer the user's query, you must only ask clarifying questions or call the `{GENERATE_PLAN_TOOL_NAME}` tool.

If the user query is already very detailed or lengthy (more than 3 sentences), do not ask for clarification and instead call the `{GENERATE_PLAN_TOOL_NAME}` tool.

For context, the date is {{current_datetime}}.

Be conversational and friendly, prefer saying "could you" rather than "I need" etc.

If you need to ask questions, follow these guidelines:
- Be concise and do not ask more than 5 questions.
- If there are ambiguous terms or questions, ask the user to clarify.
- Your questions should be a numbered list for clarity.
- Respond in the same language as the user's query.
- Make sure to gather all the information needed to carry out the research task in a concise, well-structured manner.
- Wrap up with a quick sentence on what the clarification will help with, it's ok to reference the user query closely here.
""".strip()

RESEARCH_PLAN_PROMPT = """
You are a research planner agent that generates the high level approach for deep research on a user query. Analyze the query carefully and break it down into main concepts and areas of exploration. \
Stick closely to the user query and stay on topic but be curious and avoid duplicate or overlapping exploration directions. \
Be sure to take into account the time sensitive aspects of the research topic and make sure to emphasize up to date information where appropriate. \
Focus on providing thorough research of the user's query over being helpful.

CRITICAL - You MUST only output the research plan for the deep research flow and nothing else, you are not responding to the user. \
Do not worry about the feasibility of the plan or access to data or tools, a different deep research flow will handle that.

For context, the date is {current_datetime}.

The research plan should be formatted as a numbered list of steps and have 6 or less individual steps.

Each step should be a standalone exploration question or topic that can be researched independently but may build on previous steps. The plan should be in the same language as the user's query.

Output only the numbered list of steps with no additional prefix or suffix.
""".strip()

RESEARCH_PLAN_REMINDER = """
Remember to only output the research plan and nothing else. Do not worry about the feasibility of the plan or data access.

Your response must only be a numbered list of steps with no additional prefix or suffix.
""".strip()

_ORCHESTRATOR_RESEARCH_AGENT_SECTION = f"""
## {RESEARCH_AGENT_TOOL_NAME}
The research task provided to the {RESEARCH_AGENT_TOOL_NAME} should be reasonably high level with a clear direction for investigation. \
It should not be a single short query, rather it should be 1 (or 2 if necessary) descriptive sentences that outline the direction of the investigation. \
The research task should be in the same language as the overall research plan.

CRITICAL - the {RESEARCH_AGENT_TOOL_NAME} only receives the task and has no additional context about the user's query, research plan, other research agents, or message history. \
You absolutely must provide all of the context needed to complete the task in the argument to the {RESEARCH_AGENT_TOOL_NAME}.

You should call the {RESEARCH_AGENT_TOOL_NAME} MANY times before completing with the {GENERATE_REPORT_TOOL_NAME} tool.

You are encouraged to call the {RESEARCH_AGENT_TOOL_NAME} in parallel if the research tasks are not dependent on each other, which is typically the case. NEVER call more than 3 {RESEARCH_AGENT_TOOL_NAME} calls in parallel.

## {GENERATE_REPORT_TOOL_NAME}
You should call the {GENERATE_REPORT_TOOL_NAME} tool if any of the following conditions are met:
- You have researched all of the relevant topics of the research plan.
- You have shifted away from the original research plan and believe that you are done.
- You have all of the information needed to thoroughly answer all aspects of the user's query.
- The last research cycle yielded minimal new information and future cycles are unlikely to yield more information.
"""

ORCHESTRATOR_PROMPT = f"""
You are an orchestrator agent for deep research. Your job is to conduct research by calling the {RESEARCH_AGENT_TOOL_NAME} tool with high level research tasks. \
This delegates the lower level research work to the {RESEARCH_AGENT_TOOL_NAME} which will provide back the results of the research.

For context, the date is {{current_datetime}}.

Before calling {GENERATE_REPORT_TOOL_NAME}, reason to double check that all aspects of the user's query have been well researched and that all key topics around the plan have been researched. \
There are cases where new discoveries from research may lead to a deviation from the original research plan.
In these cases, ensure that the new directions are thoroughly investigated prior to calling {GENERATE_REPORT_TOOL_NAME}.

NEVER output normal response tokens, you must only call tools.

# Tools
You have currently used {{current_cycle_count}} of {{max_cycles}} max research cycles. You do not need to use all cycles.
{_ORCHESTRATOR_RESEARCH_AGENT_SECTION}
## {THINK_TOOL_NAME}
CRITICAL - use the {THINK_TOOL_NAME} to reason between every call to the {RESEARCH_AGENT_TOOL_NAME} and before calling {GENERATE_REPORT_TOOL_NAME}. You should treat this as chain-of-thought reasoning to think deeply on what to do next. \
Be curious, identify knowledge gaps and consider new potential directions of research. Use paragraph format, do not use bullet points or lists.

NEVER use the {THINK_TOOL_NAME} in parallel with other {RESEARCH_AGENT_TOOL_NAME} or {GENERATE_REPORT_TOOL_NAME}.

Before calling {GENERATE_REPORT_TOOL_NAME}, double check that all aspects of the user's query have been researched and that all key topics around the plan have been researched (unless you have gone in a different direction).

# Research Plan
{{research_plan}}
""".strip()

ORCHESTRATOR_PROMPT_REASONING = f"""
You are an orchestrator agent for deep research. Your job is to conduct research by calling the {RESEARCH_AGENT_TOOL_NAME} tool with high level research tasks. \
This delegates the lower level research work to the {RESEARCH_AGENT_TOOL_NAME} which will provide back the results of the research.

For context, the date is {{current_datetime}}.

Before calling {GENERATE_REPORT_TOOL_NAME}, reason to double check that all aspects of the user's query have been well researched and that all key topics around the plan have been researched.
There are cases where new discoveries from research may lead to a deviation from the original research plan. In these cases, ensure that the new directions are thoroughly investigated prior to calling {GENERATE_REPORT_TOOL_NAME}.

Between calls, think deeply on what to do next. Be curious, identify knowledge gaps and consider new potential directions of research. Use paragraph format for your reasoning, do not use bullet points or lists.

NEVER output normal response tokens, you must only call tools.

# Tools
You have currently used {{current_cycle_count}} of {{max_cycles}} max research cycles. You do not need to use all cycles.
{_ORCHESTRATOR_RESEARCH_AGENT_SECTION}
# Research Plan
{{research_plan}}
""".strip()

FIRST_CYCLE_REMINDER = """
Make sure all parts of the user question and the plan have been thoroughly explored before calling generate_report. If new interesting angles have been revealed from the research, you may deviate from the plan to research new directions.
""".strip()

FINAL_REPORT_PROMPT = """
You are the final answer generator for a deep research task. Your job is to produce a thorough, balanced, and comprehensive answer on the research question provided by the user. \
You have access to high-quality, diverse sources collected by secondary research agents as well as their analysis of the sources.

IMPORTANT - You get straight to the point, never providing a title and avoiding lengthy introductions/preambles.

For context, the date is {current_datetime}.

Users have explicitly selected the deep research mode and will expect a long and detailed answer. It is ok and encouraged that your response is several pages long. \
Structure your response logically into relevant sections. You may find it helpful to reference the research plan to help structure your response but do not limit yourself to what is contained in the plan.

You use different text styles and formatting to make the response easier to read. You may use markdown rarely when necessary to make the response more digestible.

Provide inline citations in the format [1], [2], [3], etc. based on the citations included by the research agents.
""".strip()

USER_FINAL_REPORT_QUERY = f"""
The original research plan is included below (use it as a helpful reference but do not limit yourself to this):
```
{{research_plan}}
```

Based on all of the context provided in the research history, provide a comprehensive, well structured, and insightful answer to the user's previous query. \
CRITICAL: be extremely thorough in your response and address all relevant aspects of the query.

Ignore the format styles of the intermediate {RESEARCH_AGENT_TOOL_NAME} reports, those are not end user facing and different from your task.

Provide inline citations in the format [1], [2], [3], etc. based on the citations included by the research agents. The citations should be just a number in a bracket, nothing additional.
""".strip()

_RESEARCH_AGENT_TOOL_TIPS = f"""

## {WEB_SEARCH_TOOL_NAME}
Use the `{WEB_SEARCH_TOOL_NAME}` tool to get search results from the web. You should use this tool to get context for your research. These should be optimized for search engines like Google. \
Use concise and specific queries and avoid merging multiple queries into one. You can call {WEB_SEARCH_TOOL_NAME} with multiple queries at once (3 max) but generally only do this when there is a clear opportunity for parallel searching. \
If you use multiple queries, ensure that the queries are related in topic but not similar such that the results would be redundant.

## {OPEN_URL_TOOL_NAME}
Use the `{OPEN_URL_TOOL_NAME}` tool to read the content of one or more URLs. Use this tool to access the contents of the most promising web pages from your searches. \
You can open many URLs at once by passing multiple URLs in the array if multiple pages seem promising. Prioritize the most promising pages and reputable sources. \
You should almost always use {OPEN_URL_TOOL_NAME} after a {WEB_SEARCH_TOOL_NAME} call{{open_url_suffix}}.
"""

RESEARCH_AGENT_PROMPT = f"""
You are a highly capable, thoughtful, and precise research agent that conducts research on a specific topic. Prefer being thorough in research over being helpful. Be curious but stay strictly on topic. \
You iteratively call the tools available to you including {{available_tools}} until you have completed your research at which point you call the {GENERATE_REPORT_TOOL_NAME} tool.

NEVER output normal response tokens, you must only call tools.

For context, the date is {{current_datetime}}.

# Tools
You have a limited number of cycles to complete your research and you do not have to use all cycles. You are on cycle {{current_cycle_count}} of {MAX_RESEARCH_CYCLES}.\
{_RESEARCH_AGENT_TOOL_TIPS.format(open_url_suffix=f" and sometimes after reasoning with the {THINK_TOOL_NAME} tool")}
## {THINK_TOOL_NAME}
CRITICAL - use the think tool after every set of searches and reads (so search, read some pages, then think and repeat). \
You MUST use the {THINK_TOOL_NAME} before calling the {WEB_SEARCH_TOOL_NAME} tool for all calls to {WEB_SEARCH_TOOL_NAME} except for the first call. \
Use the {THINK_TOOL_NAME} before calling the {GENERATE_REPORT_TOOL_NAME} tool.

After a set of searches + reads, use the {THINK_TOOL_NAME} to analyze the results and plan the next steps.
- Reflect on the key information found with relation to the task.
- Reason thoroughly about what could be missing, the knowledge gaps, and what queries might address them, \
or why there is enough information to answer the research task comprehensively.

## {GENERATE_REPORT_TOOL_NAME}
Once you have completed your research, call the `{GENERATE_REPORT_TOOL_NAME}` tool. \
You should only call this tool after you have fully researched the topic. \
Consider other potential areas of research and weigh that against the materials already gathered before calling this tool.
""".strip()

RESEARCH_AGENT_PROMPT_REASONING = f"""
You are a highly capable, thoughtful, and precise research agent that conducts research on a specific topic. Prefer being thorough in research over being helpful. Be curious but stay strictly on topic. \
You iteratively call the tools available to you including {{available_tools}} until you have completed your research at which point you call the {GENERATE_REPORT_TOOL_NAME} tool. Between calls, think about the results of the previous tool call and plan the next steps. \
Reason thoroughly about what could be missing, identify knowledge gaps, and what queries might address them. Or consider why there is enough information to answer the research task comprehensively.

Once you have completed your research, call the `{GENERATE_REPORT_TOOL_NAME}` tool.

NEVER output normal response tokens, you must only call tools.

For context, the date is {{current_datetime}}.

# Tools
You have a limited number of cycles to complete your research and you do not have to use all cycles. You are on cycle {{current_cycle_count}} of {MAX_RESEARCH_CYCLES}.\
{_RESEARCH_AGENT_TOOL_TIPS.format(open_url_suffix="")}
## {GENERATE_REPORT_TOOL_NAME}
Once you have completed your research, call the `{GENERATE_REPORT_TOOL_NAME}` tool. You should only call this tool after you have fully researched the topic.
""".strip()

OPEN_URL_REMINDER_RESEARCH_AGENT = f"""
Remember that after using {WEB_SEARCH_TOOL_NAME}, you are encouraged to open some pages to get more context unless the query is completely answered by the snippets.
Open the pages that look the most promising and high quality by calling the {OPEN_URL_TOOL_NAME} tool with an array of URLs.
""".strip()

RESEARCH_REPORT_PROMPT = """
You are a highly capable and precise research sub-agent that has conducted research on a specific topic. \
Your job is now to organize the findings to return a comprehensive report that preserves all relevant statements and information that has been gathered in the existing messages. \
The report will be seen by another agent instead of a user so keep it free of formatting or commentary and instead focus on the facts only. \
Do not give it a title, do not break it down into sections, and do not provide any of your own conclusions/analysis.

You may see a list of tool calls in the history but you do not have access to tools anymore. You should only use the information in the history to create the report.

CRITICAL - This report should be as long as necessary to return ALL of the information that the researcher has gathered. It should be several pages long so as to capture as much detail as possible from the research. \
It cannot be stressed enough that this report must be EXTREMELY THOROUGH and COMPREHENSIVE. Only this report is going to be returned, so it's CRUCIAL that you don't lose any details from the raw messages.

Remove any obviously irrelevant or duplicative information.

If a statement seems not trustworthy or is contradictory to other statements, it is important to flag it.

Write the report in the same language as the provided task.

Cite all sources INLINE using the format [1], [2], [3], etc. based on the `document` field of the source. \
Cite inline as opposed to leaving all citations until the very end of the response.
""".strip()

USER_REPORT_QUERY = """
Please write me a comprehensive report on the research topic given the context above. As a reminder, the original topic was:
{research_topic}

Remember to include AS MUCH INFORMATION AS POSSIBLE and as faithful to the original sources as possible. \
Keep it free of formatting and focus on the facts only. Be sure to include all context for each fact to avoid misinterpretation or misattribution. \
Respond in the same language as the topic provided above.

Cite every fact INLINE using the format [1], [2], [3], etc. based on the `document` field of the source.

CRITICAL - BE EXTREMELY THOROUGH AND COMPREHENSIVE, YOUR RESPONSE SHOULD BE SEVERAL PAGES LONG.
""".strip()
# ruff: noqa: E501 end


def _function_tool(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> ToolDefinition:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


GENERATE_PLAN_TOOL = _function_tool(
    GENERATE_PLAN_TOOL_NAME,
    "No clarification needed, generate a research plan for the user's query.",
    {},
    [],
)
RESEARCH_AGENT_TOOL = _function_tool(
    RESEARCH_AGENT_TOOL_NAME,
    "Conduct research on a specific topic.",
    {
        RESEARCH_AGENT_TASK_KEY: {
            "type": "string",
            "description": "The research task to investigate, should be 1-2 descriptive sentences outlining the direction of investigation.",
        }
    },
    [RESEARCH_AGENT_TASK_KEY],
)
GENERATE_REPORT_TOOL = _function_tool(
    GENERATE_REPORT_TOOL_NAME,
    "Generate the final research report from all of the findings. Should be called when all aspects of the user's query have been researched, or maximum cycles are reached.",
    {},
    [],
)
THINK_TOOL = _function_tool(
    THINK_TOOL_NAME,
    "Use this for reasoning between research_agent calls and before calling generate_report. Think deeply about key results, identify knowledge gaps, and plan next steps.",
    {
        "reasoning": {
            "type": "string",
            "description": "Your chain of thought reasoning, use paragraph format, no lists.",
        }
    },
    ["reasoning"],
)
WEB_SEARCH_TOOL = _function_tool(
    WEB_SEARCH_TOOL_NAME,
    "Search the web. Returns numbered documents with title, url and snippet.",
    {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": f"One or more search queries ({MAX_SEARCH_QUERIES_PER_CALL} max).",
        }
    },
    ["queries"],
)
OPEN_URL_TOOL = _function_tool(
    OPEN_URL_TOOL_NAME,
    "Open web pages and return their text content as numbered documents.",
    {
        "urls": {
            "type": "array",
            "items": {"type": "string"},
            "description": f"URLs to open ({MAX_URLS_PER_CALL} max).",
        }
    },
    ["urls"],
)


def get_current_datetime() -> str:
    return datetime.now().strftime("%A %B %d, %Y")


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


class Console:
    """Progress goes to stderr so stdout carries only the clarification or report."""

    def __init__(self, quiet: bool) -> None:
        self._quiet = quiet
        self._lock = threading.Lock()

    def log(self, prefix: str, message: str) -> None:
        if self._quiet:
            return
        with self._lock:
            print(f"[{prefix}] {message}", file=sys.stderr, flush=True)

    def stream(self, text: str) -> None:
        if self._quiet:
            return
        with self._lock:
            print(text, end="", file=sys.stderr, flush=True)


def _shorten(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# LLM access
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.args)},
        }


@dataclass
class StepResult:
    content: str | None
    tool_calls: list[ToolCall]


def _parse_tool_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class LLMClient:
    def __init__(self, model: str, is_reasoning_model: bool | None) -> None:
        self.model = model
        self.is_reasoning_model = (
            is_reasoning_model
            if is_reasoning_model is not None
            else self._detect_reasoning_model(model)
        )
        self.max_input_tokens = self._lookup_max_input_tokens(model)

    @staticmethod
    def _detect_reasoning_model(model: str) -> bool:
        try:
            return bool(litellm.supports_reasoning(model=model))
        except Exception:
            return False

    @staticmethod
    def _lookup_max_input_tokens(model: str) -> int | None:
        try:
            value = litellm.get_model_info(model).get("max_input_tokens")
        except Exception:
            return None
        return int(value) if value else None

    def tool_step(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        tool_choice: str,
    ) -> StepResult:
        kwargs: dict[str, Any] = {}
        if not self.is_reasoning_model:
            kwargs["max_tokens"] = TOOL_STEP_MAX_TOKENS
        response = litellm.completion(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            timeout=LLM_STEP_TIMEOUT_S,
            **kwargs,
        )
        message = response.choices[0].message
        tool_calls = [
            ToolCall(
                id=tool_call.id or f"call_{uuid.uuid4().hex[:24]}",
                name=tool_call.function.name,
                args=_parse_tool_args(tool_call.function.arguments),
            )
            for tool_call in (message.tool_calls or [])
        ]
        return StepResult(content=message.content, tool_calls=tool_calls)

    def stream_text(
        self,
        messages: list[Message],
        max_tokens: int | None,
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = litellm.completion(
            model=self.model,
            messages=messages,
            stream=True,
            timeout=LLM_REPORT_TIMEOUT_S,
            **kwargs,
        )
        parts: list[str] = []
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                parts.append(delta)
                if on_delta:
                    on_delta(delta)
        return "".join(parts)


def system_message(content: str) -> Message:
    return {"role": "system", "content": content}


def user_message(content: str) -> Message:
    return {"role": "user", "content": content}


def assistant_tool_call_message(tool_calls: list[ToolCall]) -> Message:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [tool_call.to_openai_format() for tool_call in tool_calls],
    }


def tool_response_message(tool_call_id: str, content: str) -> Message:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def last_n_user_messages(history: list[Message], n: int) -> list[Message]:
    """Keep everything from the n-th most recent user message onward."""
    seen = 0
    for index in range(len(history) - 1, -1, -1):
        if history[index]["role"] == "user":
            seen += 1
            if seen == n:
                return history[index:]
    return history


def find_tool_call(tool_calls: list[ToolCall], name: str) -> ToolCall | None:
    return next((tc for tc in tool_calls if tc.name == name), None)


# ---------------------------------------------------------------------------
# Web tools and citations
# ---------------------------------------------------------------------------


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class Source:
    title: str
    url: str


CitationMap = dict[int, Source]


class WebSearchProvider(Protocol):
    def search(self, query: str, max_results: int) -> list[SearchHit]: ...


class DuckDuckGoSearch:
    """Keyless web search. Swap in another WebSearchProvider (Exa, Serper, ...) for better results."""

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        from ddgs import DDGS

        hits = DDGS().text(query, max_results=max_results)
        return [
            SearchHit(
                title=hit.get("title", ""),
                url=hit.get("href", ""),
                snippet=hit.get("body", ""),
            )
            for hit in hits
            if hit.get("href")
        ]


def fetch_page_text(url: str) -> tuple[str, str]:
    """Returns (title, text) for an HTML or plain-text page."""
    response = httpx.get(
        url,
        follow_redirects=True,
        timeout=URL_FETCH_TIMEOUT_S,
        headers={"User-Agent": "Mozilla/5.0 (compatible; deep-research-agent/1.0)"},
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and not content_type.startswith("text/"):
        raise ValueError(f"Unsupported content type: {content_type or 'unknown'}")
    if "html" not in content_type:
        return url, response.text[:MAX_PAGE_CHARS]

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else url
    for tag in soup(
        ["script", "style", "noscript", "nav", "footer", "header", "form", "svg"]
    ):
        tag.decompose()
    lines = (line.strip() for line in soup.get_text("\n").splitlines())
    text = "\n".join(line for line in lines if line)
    return title, text[:MAX_PAGE_CHARS]


_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class CitationRegistry:
    """Global source numbering shared across all research agents of one run."""

    def __init__(self) -> None:
        self._number_by_url: dict[str, int] = {}
        self.sources: CitationMap = {}

    def add(self, source: Source) -> int:
        existing = self._number_by_url.get(source.url)
        if existing is not None:
            return existing
        number = len(self.sources) + 1
        self._number_by_url[source.url] = number
        self.sources[number] = source
        return number

    def collapse(self, text: str, local_citations: CitationMap) -> str:
        """Renumber a sub-agent report's local [n] markers into global numbers."""

        def replace(match: re.Match[str]) -> str:
            numbers = [int(raw) for raw in match.group(1).split(",")]
            sources = [local_citations[n] for n in numbers if n in local_citations]
            if not sources:
                return match.group(0)
            return "".join(f"[{self.add(source)}]" for source in sources)

        return _CITATION_PATTERN.sub(replace, text)

    def cited_in(self, text: str) -> CitationMap:
        cited: CitationMap = {}
        for match in _CITATION_PATTERN.finditer(text):
            for raw in match.group(1).split(","):
                number = int(raw)
                if number in self.sources:
                    cited[number] = self.sources[number]
        return dict(sorted(cited.items()))


# ---------------------------------------------------------------------------
# Research sub-agent (run_research_agent_call)
# ---------------------------------------------------------------------------


@dataclass
class ResearchAgentResult:
    report: str
    citations: CitationMap


@dataclass
class ResearchAgent:
    task: str
    label: str
    llm: LLMClient
    search_provider: WebSearchProvider
    console: Console
    citations: CitationMap = field(default_factory=dict)
    _number_by_url: dict[str, int] = field(default_factory=dict)
    # open_url fetches pages concurrently, all of which register citations.
    _cite_lock: threading.Lock = field(default_factory=threading.Lock)

    def _cite(self, title: str, url: str) -> int:
        with self._cite_lock:
            existing = self._number_by_url.get(url)
            if existing is not None:
                return existing
            number = len(self.citations) + 1
            self._number_by_url[url] = number
            self.citations[number] = Source(title=title or url, url=url)
            return number

    def _web_search(self, args: dict[str, Any]) -> tuple[str, bool]:
        queries = [str(q) for q in args.get("queries") or []][
            :MAX_SEARCH_QUERIES_PER_CALL
        ]
        if not queries:
            raise ValueError("web_search requires a non-empty 'queries' list")
        results: list[dict[str, Any]] = []
        for query in queries:
            self.console.log(self.label, f"search: {query}")
            try:
                hits = self.search_provider.search(query, SEARCH_RESULTS_PER_QUERY)
            except Exception as e:
                results.append({"query": query, "error": str(e)})
                continue
            for hit in hits:
                results.append(
                    {
                        "document": self._cite(hit.title, hit.url),
                        "title": hit.title,
                        "url": hit.url,
                        "snippet": hit.snippet,
                    }
                )
        found_any = any("document" in result for result in results)
        return json.dumps({"results": results}, ensure_ascii=False), found_any

    def _open_urls(self, args: dict[str, Any]) -> str:
        urls = [str(u) for u in args.get("urls") or []][:MAX_URLS_PER_CALL]
        if not urls:
            raise ValueError("open_url requires a non-empty 'urls' list")
        for url in urls:
            self.console.log(self.label, f"open: {url}")

        def fetch(url: str) -> dict[str, Any]:
            try:
                title, text = fetch_page_text(url)
            except Exception as e:
                return {"url": url, "error": str(e)}
            return {
                "document": self._cite(title, url),
                "title": title,
                "url": url,
                "content": text,
            }

        with ThreadPoolExecutor(max_workers=len(urls)) as executor:
            pages = list(executor.map(fetch, urls))
        return json.dumps({"results": pages}, ensure_ascii=False)

    def _system_prompt(self, cycle: int) -> str:
        template = (
            RESEARCH_AGENT_PROMPT_REASONING
            if self.llm.is_reasoning_model
            else RESEARCH_AGENT_PROMPT
        )
        return template.format(
            available_tools=f"{WEB_SEARCH_TOOL_NAME} and {OPEN_URL_TOOL_NAME}",
            current_datetime=get_current_datetime(),
            current_cycle_count=cycle,
        )

    def _tool_definitions(self) -> list[ToolDefinition]:
        tools = [WEB_SEARCH_TOOL, OPEN_URL_TOOL, GENERATE_REPORT_TOOL]
        if not self.llm.is_reasoning_model:
            tools.append(THINK_TOOL)
        return tools

    def run(self) -> ResearchAgentResult:
        self.console.log(self.label, f"task: {_shorten(self.task, 300)}")
        start_time = time.monotonic()
        history: list[Message] = [user_message(self.task)]
        research_cycle_count = 0
        just_ran_web_search = False

        while research_cycle_count < MAX_RESEARCH_CYCLES:
            if time.monotonic() - start_time > RESEARCH_AGENT_FORCE_REPORT_SECONDS:
                self.console.log(self.label, "time limit hit, forcing report")
                break

            messages = [
                system_message(self._system_prompt(research_cycle_count)),
                *history,
            ]
            if just_ran_web_search:
                messages.append(user_message(OPEN_URL_REMINDER_RESEARCH_AGENT))

            step = self.llm.tool_step(messages, self._tool_definitions(), "required")
            # Only one tool type is executed per step, matching Onyx.
            tool_calls = (
                [tc for tc in step.tool_calls if tc.name == step.tool_calls[0].name]
                if step.tool_calls
                else []
            )
            just_ran_web_search = False

            if find_tool_call(tool_calls, GENERATE_REPORT_TOOL_NAME):
                break

            think_call = find_tool_call(tool_calls, THINK_TOOL_NAME)
            if think_call:
                # Think steps do not consume a research cycle.
                self.console.log(
                    self.label,
                    f"think: {_shorten(str(think_call.args.get('reasoning', '')))}",
                )
                history.append(assistant_tool_call_message([think_call]))
                history.append(
                    tool_response_message(think_call.id, THINK_TOOL_RESPONSE_MESSAGE)
                )
                continue

            if tool_calls:
                history.append(assistant_tool_call_message(tool_calls))
                for tool_call in tool_calls:
                    try:
                        if tool_call.name == WEB_SEARCH_TOOL_NAME:
                            response, found_results = self._web_search(tool_call.args)
                            just_ran_web_search = just_ran_web_search or found_results
                        elif tool_call.name == OPEN_URL_TOOL_NAME:
                            response = self._open_urls(tool_call.args)
                        else:
                            raise ValueError(f"Unknown tool '{tool_call.name}'")
                    except Exception as e:
                        response = f"Tool call failed: {e}"
                    history.append(tool_response_message(tool_call.id, response))

            research_cycle_count += 1

        return ResearchAgentResult(
            report=self._generate_intermediate_report(history),
            citations=self.citations,
        )

    def _generate_intermediate_report(self, history: list[Message]) -> str:
        self.console.log(self.label, "writing intermediate report")
        messages = [
            system_message(RESEARCH_REPORT_PROMPT),
            *history,
            user_message(USER_REPORT_QUERY.format(research_topic=self.task)),
        ]
        report = self.llm.stream_text(messages, MAX_INTERMEDIATE_REPORT_LENGTH_TOKENS)
        if not report:
            raise ValueError(
                f"LLM failed to generate a report for research task: {self.task}"
            )
        self.console.log(self.label, f"done ({len(report)} chars)")
        return report


def run_research_agents(
    research_calls: list[ToolCall],
    llm: LLMClient,
    search_provider: WebSearchProvider,
    registry: CitationRegistry,
    console: Console,
    agent_offset: int,
) -> list[str | None]:
    """Run sub-agents in parallel; returns one report per call (None on failure)."""
    agents = [
        ResearchAgent(
            task=str(call.args.get(RESEARCH_AGENT_TASK_KEY, "")),
            label=f"agent {agent_offset + index + 1}",
            llm=llm,
            search_provider=search_provider,
            console=console,
        )
        for index, call in enumerate(research_calls)
    ]

    executor = ThreadPoolExecutor(max_workers=len(agents))
    futures = [executor.submit(agent.run) for agent in agents]
    wait(futures, timeout=RESEARCH_AGENT_TIMEOUT_SECONDS)
    # Like Onyx, a timed-out agent thread is abandoned rather than killed.
    executor.shutdown(wait=False, cancel_futures=True)

    reports: list[str | None] = []
    for agent, future in zip(agents, futures):
        if not future.done():
            console.log(agent.label, "timed out")
            reports.append(RESEARCH_AGENT_TIMEOUT_MESSAGE)
            continue
        error = future.exception()
        if error is not None:
            console.log(agent.label, f"failed: {error}")
            reports.append(None)
            continue
        result = future.result()
        reports.append(registry.collapse(result.report, result.citations))
    return reports


# ---------------------------------------------------------------------------
# Deep research loop (run_deep_research_llm_loop)
# ---------------------------------------------------------------------------


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
    console.log("orchestrator", f"cycle {cycle}: think: {_shorten(reasoning, 400)}")
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
