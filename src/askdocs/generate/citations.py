"""Post-generation citation validation: the hard gate between the LLM and the user.

For every claim we check:
  1. it cites at least one chunk,
  2. every cited ID is one of the chunks we actually retrieved (catches hallucinated IDs),
  3. the cited text actually supports the claim:
       - "lexical": share of the claim's content words present in the cited chunks (free, fast)
       - "llm":     an LLM judge reads claim + cited text and returns supported/unsupported
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from askdocs.generate.providers import LLMProvider, Usage
from askdocs.generate.schemas import DraftClaim, SupportVerdicts

_STOP = set(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "into",
        "over",
        "under",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "can",
        "could",
        "may",
        "might",
        "must",
        "should",
        "would",
        "will",
        "shall",
        "do",
        "does",
        "did",
        "not",
        "no",
        "yes",
        "you",
        "your",
        "they",
        "their",
        "them",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "why",
        "how",
        "than",
        "so",
        "such",
        "also",
        "only",
        "just",
        "more",
        "most",
        "other",
        "some",
        "any",
        "each",
        "all",
        "both",
        "either",
        "neither",
        "there",
        "here",
        "use",
        "used",
        "using",
    ]
)


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9][a-z0-9._/-]*", text.lower())
    return {w.strip("._/-") for w in words if w not in _STOP and len(w) > 2}


def lexical_support(claim: str, evidence: str) -> float:
    cw = content_words(claim)
    if not cw:
        return 1.0
    ev = content_words(evidence)
    return len(cw & ev) / len(cw)


@dataclass
class ClaimCheck:
    claim: DraftClaim
    ok: bool
    problems: list[str] = field(default_factory=list)
    support: float | None = None


@dataclass
class ValidationReport:
    checks: list[ClaimCheck]
    usage: Usage = field(default_factory=Usage)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    @property
    def pass_rate(self) -> float:
        return sum(c.ok for c in self.checks) / len(self.checks) if self.checks else 0.0

    def feedback(self) -> str:
        lines = []
        for i, c in enumerate(self.checks):
            if not c.ok:
                lines.append(f"- claim {i} ({c.claim.text[:80]!r}): " + "; ".join(c.problems))
        return "\n".join(lines)


SUPPORT_SYSTEM = (
    "You verify citations. For each claim, decide whether the cited evidence text directly states or clearly "
    "implies the claim. Paraphrase is fine; extra information not in the evidence is NOT supported."
)


def validate(
    claims: list[DraftClaim],
    evidence: dict[str, str],
    mode: str = "lexical",
    threshold: float = 0.5,
    judge: LLMProvider | None = None,
) -> ValidationReport:
    checks: list[ClaimCheck] = []
    for claim in claims:
        problems: list[str] = []
        if not claim.chunk_ids:
            problems.append("no citation")
        unknown = [cid for cid in claim.chunk_ids if cid not in evidence]
        if unknown:
            problems.append(f"cites unknown chunk ids {unknown}")
        checks.append(ClaimCheck(claim=claim, ok=not problems, problems=problems))

    report = ValidationReport(checks)
    structurally_ok = [c for c in checks if c.ok]
    if not structurally_ok:
        return report

    if mode == "llm" and judge is not None:
        blocks = []
        for i, c in enumerate(structurally_ok):
            ev = "\n---\n".join(evidence[cid] for cid in c.claim.chunk_ids)
            blocks.append(f'<claim index="{i}">\n{c.claim.text}\n<evidence>\n{ev}\n</evidence>\n</claim>')
        verdicts, usage = judge.structured(SUPPORT_SYSTEM, "\n\n".join(blocks), SupportVerdicts, max_tokens=1500)
        report.usage.add(usage)
        by_idx = {v.claim_index: v for v in verdicts.verdicts}
        for i, c in enumerate(structurally_ok):
            v = by_idx.get(i)
            if v is None or not v.supported:
                c.ok = False
                c.problems.append("cited text does not support the claim" + (f" ({v.reason})" if v else ""))
    else:
        for c in structurally_ok:
            ev = " ".join(evidence[cid] for cid in c.claim.chunk_ids)
            c.support = lexical_support(c.claim.text, ev)
            if c.support < threshold:
                c.ok = False
                c.problems.append(f"cited text does not support the claim (overlap {c.support:.2f} < {threshold})")
    return report
