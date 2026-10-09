"""check_citations.py - checks a report against its sources.json.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It exits 0 and prints "OK: ..." when the report is consistent, else prints each problem and exits 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")  # [3]  [1, 2]  [1-3]; not [3](link)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REF_LINE = re.compile(r"^\[(\d+)\]")
_URL = re.compile(r"https?://[^\s)\]>]+")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]
    problems = []
    by_n, seen_urls = {}, set()
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"source entry is not an object: {entry!r}")
            continue
        n, url = entry.get("n"), entry.get("url")
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"source n={n!r} is not an integer")
            continue
        if n in by_n:
            problems.append(f"source number [{n}] appears twice in sources.json")
        by_n[n] = url
        if not isinstance(url, str) or not re.match(r"https?://", url):
            problems.append(f"source [{n}] url must start with http:// or https://: {url!r}")
        elif url in seen_urls:
            problems.append(f"source [{n}] duplicates a url already used: {url}")
        else:
            seen_urls.add(url)

    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("missing '## References' heading")
        return problems
    body = report_text[:matches[-1].start()]
    refs = report_text[matches[-1].end():]

    cited = set()
    for i, segment in enumerate(_CODE.split(body)):
        if i % 2:
            continue  # code span / block: not a citation
        for m in _GROUP.finditer(segment):
            cited.update(_group_numbers(m.group(1)))
    for n in sorted(cited - by_n.keys()):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(by_n.keys() - cited):
        problems.append(f"source [{n}] never cited in the body")

    ref_numbers = []
    for line in refs.splitlines():
        line = line.strip()
        m = _REF_LINE.match(line)
        if not m:
            continue
        n = int(m.group(1))
        ref_numbers.append(n)
        if n not in by_n:
            problems.append(f"reference line [{n}] is not a source")
            continue
        urls = _URL.findall(line)
        if len(urls) != 1:
            problems.append(f"reference [{n}] must contain exactly one URL, found {len(urls)}")
        elif urls[0] != by_n[n]:
            problems.append(f"reference [{n}] url {urls[0]} does not match sources.json ({by_n[n]})")
    for n in sorted(set(ref_numbers)):
        if ref_numbers.count(n) > 1:
            problems.append(f"reference [{n}] appears more than once")
    for n in sorted(by_n.keys() - set(ref_numbers)):
        problems.append(f"source [{n}] has no line in ## References")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
