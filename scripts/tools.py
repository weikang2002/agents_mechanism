"""Tool definitions (JSON schemas) exposed to the orchestrator and research sub-agents."""

from typing import Any

from prompts import (
    GENERATE_PLAN_TOOL_NAME,
    GENERATE_REPORT_TOOL_NAME,
    OPEN_URL_TOOL_NAME,
    RESEARCH_AGENT_TASK_KEY,
    RESEARCH_AGENT_TOOL_NAME,
    THINK_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
)
from web_tools import MAX_SEARCH_QUERIES_PER_CALL, MAX_URLS_PER_CALL

ToolDefinition = dict[str, Any]


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
    name=GENERATE_PLAN_TOOL_NAME,
    description="No clarification needed, generate a research plan for the user's query.",
    properties={},
    required=[],
)

RESEARCH_AGENT_TOOL = _function_tool(
    name=RESEARCH_AGENT_TOOL_NAME,
    description="Conduct research on a specific topic.",
    properties={
        RESEARCH_AGENT_TASK_KEY: {
            "type": "string",
            "description": "The research task to investigate, should be 1-2 descriptive sentences outlining the direction of investigation.",
        }
    },
    required=[RESEARCH_AGENT_TASK_KEY],
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
