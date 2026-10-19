"""Web search, page fetching and global citation numbering."""

import re
from dataclasses import dataclass
from typing import Protocol

import httpx
from bs4 import BeautifulSoup

MAX_SEARCH_QUERIES_PER_CALL = 3
SEARCH_RESULTS_PER_QUERY = 8
MAX_URLS_PER_CALL = 5
MAX_PAGE_CHARS = 20000
URL_FETCH_TIMEOUT_S = 20


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
