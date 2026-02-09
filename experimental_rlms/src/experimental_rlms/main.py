#!/usr/bin/env python
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from experimental_rlms.crews.research_crew.research_crew import ResearchCrew

DEFAULT_URLS = [
    "https://arxiv.org/pdf/2503.09572",
    "https://research.trychroma.com/context-rot",
]


def _detect_content_type(url: str, headers: dict[str, str]) -> str:
    """Return 'pdf' or 'html' based on response headers and URL."""
    ct = headers.get("Content-Type", "").lower()
    if "application/pdf" in ct or url.rstrip("/").endswith(".pdf"):
        return "pdf"
    return "html"


def _extract_pdf(data: bytes) -> str:
    """Extract text from PDF bytes using pymupdf."""
    import pymupdf  # type: ignore[import-untyped]

    with TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "paper.pdf"
        pdf_path.write_bytes(data)
        doc = pymupdf.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    return text


def _extract_html(data: bytes, encoding: str = "utf-8") -> str:
    """Convert HTML to markdown using html2text."""
    import html2text  # type: ignore[import-untyped]

    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0  # no wrapping
    return h.handle(data.decode(encoding, errors="replace"))


def _title_from_url(url: str) -> str:
    """Derive a short title from a URL."""
    # Strip trailing slashes and take last path segment
    path = url.rstrip("/").split("/")[-1]
    # Remove file extensions
    for ext in (".pdf", ".html", ".htm"):
        if path.lower().endswith(ext):
            path = path[: -len(ext)]
    return path or url


# ---------------------------------------------------------------------------
# Research Paper Analysis Flow
# ---------------------------------------------------------------------------


class ResearchState(BaseModel):
    urls: list[str] = []
    topic: str = ""
    papers_text: dict[str, str] = {}
    literature_review: str = ""


class ResearchFlow(Flow[ResearchState]):

    @start()
    def set_sources(self):
        if not self.state.urls:
            self.state.urls = list(DEFAULT_URLS)
        print(f"Sources ({len(self.state.urls)}):")
        for url in self.state.urls:
            print(f"  - {url}")
        if self.state.topic:
            print(f"Topic: {self.state.topic}")

    @listen(set_sources)
    def fetch_and_extract(self):
        """Fetch each URL, detect content type, and extract text."""
        papers_text: dict[str, str] = {}

        for url in self.state.urls:
            title = _title_from_url(url)
            print(f"  Fetching: {url}")
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (research-flow)"},
                )
                with urllib.request.urlopen(req) as resp:  # noqa: S310
                    resp_headers = {k: v for k, v in resp.headers.items()}
                    data = resp.read()
            except Exception as exc:
                print(f"    Fetch failed: {exc}")
                continue

            content_type = _detect_content_type(url, resp_headers)
            print(f"    Detected: {content_type}")

            try:
                if content_type == "pdf":
                    text = _extract_pdf(data)
                else:
                    encoding = "utf-8"
                    ct = resp_headers.get("Content-Type", "")
                    if "charset=" in ct:
                        encoding = ct.split("charset=")[-1].split(";")[0].strip()
                    text = _extract_html(data, encoding)

                papers_text[title] = text
                print(f"    Extracted {len(text):,} characters")
            except Exception as exc:
                print(f"    Extraction failed: {exc}")

        self.state.papers_text = papers_text
        total_chars = sum(len(t) for t in papers_text.values())
        print(f"Total: {total_chars:,} characters across {len(papers_text)} sources")

    @listen(fetch_and_extract)
    def analyze_papers(self):
        """Run the RLM-powered research crew over all extracted texts."""
        if not self.state.papers_text:
            print("No content to analyze — skipping.")
            self.state.literature_review = "No content was available for analysis."
            return

        paper_titles = ", ".join(self.state.papers_text.keys())
        print(f"Analyzing {len(self.state.papers_text)} sources with RLM crew...")

        topic = self.state.topic or "research analysis"
        research_crew = ResearchCrew()
        research_crew.papers_dict = self.state.papers_text
        result = (
            research_crew
            .crew()
            .kickoff(inputs={
                "topic": topic,
                "paper_titles": paper_titles,
            })
        )
        self.state.literature_review = result.raw

    @listen(analyze_papers)
    def save_review(self):
        """Write the literature review to a markdown file."""
        topic = self.state.topic or "Research Analysis"
        header_lines = [
            f"# Literature Review: {topic}",
            "",
            f"**Sources analyzed:** {len(self.state.papers_text)}",
            "",
            "| Source | Characters |",
            "| ------ | ---------- |",
        ]
        for title, text in self.state.papers_text.items():
            header_lines.append(f"| {title} | {len(text):,} |")
        header_lines += ["", "---", ""]

        content = "\n".join(header_lines) + self.state.literature_review

        out_path = Path("literature_review.md")
        out_path.write_text(content)
        print(f"Literature review saved to {out_path.resolve()}")


def research():
    """Entry point: accepts URLs as positional args, optional --topic flag.

    Usage:
        uv run research URL1 URL2 ...
        uv run research --topic "context windows" URL1 URL2
        uv run research   # uses default URLs
    """
    args = sys.argv[1:]
    topic = ""
    urls: list[str] = []

    i = 0
    while i < len(args):
        if args[i] == "--topic" and i + 1 < len(args):
            topic = args[i + 1]
            i += 2
        else:
            urls.append(args[i])
            i += 1

    inputs: dict[str, object] = {}
    if urls:
        inputs["urls"] = urls
    if topic:
        inputs["topic"] = topic

    flow = ResearchFlow()
    flow.kickoff(inputs=inputs)


if __name__ == "__main__":
    research()
