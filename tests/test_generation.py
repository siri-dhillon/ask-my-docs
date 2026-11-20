from askdocs.generate.answer import IDK, generate_answer
from askdocs.generate.citations import lexical_support, validate
from askdocs.generate.fake_handlers import default_fake
from askdocs.generate.schemas import AnswerDraft, DraftClaim, SupportVerdict, SupportVerdicts
from askdocs.models import RetrievedChunk
from askdocs.service import AskService
from tests.conftest import CORPUS

EVIDENCE = {c.id: c.text for c in CORPUS}


def rc(i: int) -> RetrievedChunk:
    return RetrievedChunk(chunk=CORPUS[i], score=0.9)


def test_lexical_support_scores_overlap():
    assert lexical_support("A ConfigMap cannot exceed 1 MiB", EVIDENCE["configmap-000001"]) > 0.7
    assert lexical_support("Pods are scheduled by etcd quorum leaders", EVIDENCE["configmap-000001"]) < 0.3


def test_validator_flags_unknown_ids_and_missing_citations():
    report = validate(
        [
            DraftClaim(text="ConfigMaps cannot exceed 1 MiB.", chunk_ids=["configmap-000001"]),
            DraftClaim(text="Secrets hold sensitive data.", chunk_ids=["made-up-999"]),
            DraftClaim(text="Uncited sentence.", chunk_ids=[]),
        ],
        EVIDENCE,
    )
    assert [c.ok for c in report.checks] == [True, False, False]
    assert "unknown" in report.checks[1].problems[0]
    assert not report.ok


def test_validator_flags_unsupported_claim():
    report = validate([DraftClaim(text="HPA scales nodes using GPU temperature.", chunk_ids=["configmap-000001"])], EVIDENCE)
    assert not report.ok and "does not support" in report.checks[0].problems[0]


def test_llm_judge_mode(settings):
    fake = default_fake()
    fake.handlers[SupportVerdicts] = lambda s, u: SupportVerdicts(
        verdicts=[SupportVerdict(claim_index=0, supported=False, reason="not stated")]
    )
    report = validate([DraftClaim(text="x", chunk_ids=["secret-000002"])], EVIDENCE, mode="llm", judge=fake)
    assert not report.ok and "not stated" in report.checks[0].problems[0]


def test_happy_path_answer_is_cited(settings):
    res = generate_answer("How big can a ConfigMap be?", [rc(0), rc(1)], default_fake(), settings)
    assert not res.refused and res.attempts == 1
    assert "[configmap-000001]" in res.answer_text


def test_retry_then_success(settings):
    fake = default_fake()
    drafts = iter(
        [
            AnswerDraft(answerable=True, claims=[DraftClaim(text="ConfigMaps max 1 MiB.", chunk_ids=["bogus-1"])]),
            AnswerDraft(
                answerable=True, claims=[DraftClaim(text="ConfigMap data cannot exceed 1 MiB.", chunk_ids=["configmap-000001"])]
            ),
        ]
    )
    fake.handlers[AnswerDraft] = lambda s, u: next(drafts)
    res = generate_answer("size?", [rc(0)], fake, settings)
    assert not res.refused and res.attempts == 2
    retry_prompt = fake.calls[-1][1]
    assert "failed citation validation" in retry_prompt and "bogus-1" in retry_prompt


def test_persistent_bad_citations_become_idk(settings):
    fake = default_fake()
    fake.handlers[AnswerDraft] = lambda s, u: AnswerDraft(
        answerable=True, claims=[DraftClaim(text="Totally unrelated statement about cats.", chunk_ids=["configmap-000001"])]
    )
    res = generate_answer("size?", [rc(0)], fake, settings)
    assert res.refused and res.answer_text == IDK and res.refusal_reason == "citation validation failed"


def test_model_says_unanswerable(settings):
    fake = default_fake()
    fake.handlers[AnswerDraft] = lambda s, u: AnswerDraft(answerable=False, claims=[])
    res = generate_answer("What's the capital of France?", [rc(0)], fake, settings)
    assert res.refused and res.attempts == 1


def test_service_end_to_end_with_fake_llm(settings, index):
    from askdocs.retrieve.pipeline import Retriever
    from tests.test_retrieval import StubReranker

    svc = AskService(settings, index=index, retriever=Retriever(settings, index, StubReranker()), llm=default_fake())
    resp = svc.ask("Liveness probes: when does the kubelet restart a container?")
    assert not resp.refused
    assert resp.citations[0].chunk_id == "probes-000003"
    assert {"retrieval_ms", "generation_ms", "total_ms"} <= resp.timings_ms.keys()

    refused = svc.ask("zzz qqq")
    assert refused.refused and "confidence" in (refused.refusal_reason or "")
