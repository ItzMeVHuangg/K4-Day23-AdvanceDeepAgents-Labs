"""research.py - The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from langchain_core.rate_limiters import InMemoryRateLimiter

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to the validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(topic).lower()).strip("-")[:60].strip("-")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Research topic: {topic}\n\n"
        "Produce a survey report in English following your workflow: plan with write_todos, delegate sub-questions to "
        "`researcher` subagents in parallel, merge the notes into sources.json, write the report body (no References section), "
        "run the finalizer, run the validator until it prints OK, then spot-check claims with `citation-checker`. "
        "Every claim must come from the researchers' notes and carry an inline [n] citation. "
        f"The final report must be at {REPORT_PATH}."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.
    Lead messages only: subagent tokens are not included, so this undercounts the real cost."""
    calls = Counter()
    tokens = {"input": 0, "output": 0}
    for message in messages:
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
        for call in tool_calls or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                calls[name] += 1
        usage = message.get("usage_metadata") if isinstance(message, dict) else getattr(message, "usage_metadata", None)
        if usage:
            tokens["input"] += int(usage.get("input_tokens") or 0)
            tokens["output"] += int(usage.get("output_tokens") or 0)
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": calls.get("task", 0),
            "tool_calls": dict(calls), "tokens": tokens}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.
    Raises RuntimeError (and writes nothing) when the report is missing/empty or sources.json is missing/invalid."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes, sources_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip():
        raise RuntimeError(f"the agent did not produce a report at {REPORT_PATH}")
    if not sources_bytes:
        raise RuntimeError(f"the agent did not produce {SOURCES_PATH}")
    try:
        sources = json.loads(sources_bytes.decode("utf-8"))
        report = report_bytes.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"unreadable output from the sandbox: {exc}") from exc
    if not isinstance(sources, list) or not all(isinstance(s, dict) for s in sources):
        raise RuntimeError("sources.json must be a JSON list of objects")

    meta = {"topic": topic, **summarize(messages, elapsed, model_name), "n_sources": len(sources),
            "source_families": sorted({str(s["source"]) for s in sources if s.get("source")})}
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    (reports_dir / f"{slug}.sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = reports_dir / f"{slug}.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = (topic or "").strip()
    if not topic:
        print('usage: python research.py "survey about world model"', file=sys.stderr)
        return 2
    try:
        model = make_model()
        # one limiter shared by the lead and every subagent (they reuse this model object): stay under the provider's RPM.
        # LAB_RPS = requests per second (default 0.2 = 12 per minute); 0 disables it.
        rps = float(os.getenv("LAB_RPS", "0.2"))
        if rps > 0:
            model.rate_limiter = InMemoryRateLimiter(requests_per_second=rps, max_bucket_size=1)
        model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or "unknown"
        start = time.monotonic()
        with open_sandbox() as backend:  # always cleaned up, even on errors
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
            agent = build_lead_agent(backend, model)
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                  config={"recursion_limit": 1000})
            path = save_outputs(backend, topic, result["messages"], time.monotonic() - start, str(model_name))
    except Exception as exc:  # noqa: BLE001 - a failed run must exit 1 with a clear message
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Report saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
