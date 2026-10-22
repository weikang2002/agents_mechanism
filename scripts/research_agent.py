"""Research sub-agent: a tool loop over web_search/open_url that ends in an intermediate report."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

from llm import (
    LLMClient,
    Message,
    ToolCall,
    assistant_tool_call_message,
    find_tool_call,
    system_message,
    tool_response_message,
    user_message,
)
from prompts import (
    GENERATE_REPORT_TOOL_NAME,
    OPEN_URL_REMINDER_RESEARCH_AGENT,
    OPEN_URL_TOOL_NAME,
    RESEARCH_AGENT_PROMPT,
    RESEARCH_AGENT_PROMPT_REASONING,
    RESEARCH_AGENT_TASK_KEY,
    RESEARCH_REPORT_PROMPT,
    THINK_TOOL_NAME,
    THINK_TOOL_RESPONSE_MESSAGE,
    USER_REPORT_QUERY,
    WEB_SEARCH_TOOL_NAME,
)
from tools import (
    GENERATE_REPORT_TOOL,
    OPEN_URL_TOOL,
    RESEARCH_AGENT_THINK_TOOL,
    WEB_SEARCH_TOOL,
    ToolDefinition,
)
from utils import Console, get_current_datetime, shorten
from web_tools import (
    MAX_SEARCH_QUERIES_PER_CALL,
    MAX_URLS_PER_CALL,
    SEARCH_RESULTS_PER_QUERY,
    CitationMap,
    CitationRegistry,
    Source,
    WebSearchProvider,
    fetch_page_text,
)

# Research sub-agent limits (tools/fake_tools/research_agent.py)
MAX_RESEARCH_CYCLES = 8
RESEARCH_AGENT_TIMEOUT_SECONDS = 30 * 60
RESEARCH_AGENT_TIMEOUT_MESSAGE = "Research Agent timed out after 30 minutes"
RESEARCH_AGENT_FORCE_REPORT_SECONDS = 12 * 60
MAX_INTERMEDIATE_REPORT_LENGTH_TOKENS = 10000
RESEARCH_AGENT_FAILURE_MESSAGE = "Research agent call failed. Try a different approach or continue without this result."


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
            max_cycles=MAX_RESEARCH_CYCLES,
        )

    def _tool_definitions(self) -> list[ToolDefinition]:
        tools = [WEB_SEARCH_TOOL, OPEN_URL_TOOL, GENERATE_REPORT_TOOL]
        if not self.llm.is_reasoning_model:
            tools.append(RESEARCH_AGENT_THINK_TOOL)
        return tools

    def run(self) -> ResearchAgentResult:
        self.console.log(self.label, f"task: {shorten(self.task, 300)}")
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
                    f"think: {shorten(str(think_call.args.get('reasoning', '')))}",
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
