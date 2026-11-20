"""Deterministic stand-ins for the LLM, used by tests and by CI when no API key is configured."""

from __future__ import annotations

import re

from askdocs.generate.providers import FakeProvider
from askdocs.generate.schemas import AnswerDraft, DraftClaim, RewrittenQuery, SupportVerdict, SupportVerdicts

_CHUNK = re.compile(r'<chunk id="([^"]+)"[^>]*>\n(.*?)\n</chunk>', re.S)


def fake_answer(system: str, user: str) -> AnswerDraft:
    """Answer with the first sentence of the first chunk, citing that chunk."""
    chunks = _CHUNK.findall(user)
    if not chunks:
        return AnswerDraft(answerable=False, claims=[])
    cid, text = chunks[0]
    sentence = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return AnswerDraft(answerable=True, claims=[DraftClaim(text=sentence, chunk_ids=[cid])])


def fake_support(system: str, user: str) -> SupportVerdicts:
    n = len(re.findall(r"<claim index=", user))
    return SupportVerdicts(verdicts=[SupportVerdict(claim_index=i, supported=True, reason="fake") for i in range(n)])


def fake_rewrite(system: str, user: str) -> RewrittenQuery:
    return RewrittenQuery(query=user.strip().splitlines()[-1])


def default_fake() -> FakeProvider:
    return FakeProvider(handlers={AnswerDraft: fake_answer, SupportVerdicts: fake_support, RewrittenQuery: fake_rewrite})
