"""Information availability, missingness, receipt clocks, correction chains and complete paging."""

from datetime import timedelta

import pytest

from coleman.corpus.stress_feedback import StressFeedbackPage, decision_time, feedback_pages, visible_feedback
from coleman.corpus.stress_replay import schedule_build
from coleman.corpus.stress_world import generate_build
from coleman.observations.contracts import ObservationSnapshot
from tests.corpus.stress_support import case


def observations(pages, cutoff):
    return tuple(row for snapshot in visible_feedback(pages, cutoff) for row in snapshot.observations)


@pytest.mark.parametrize("mode", ["full", "censored", "delayed", "missing", "corrected"])
def test_feedback_modes_never_expose_current_truth_at_decision(mode):
    value = case(feedback=mode)
    pages = tuple(feedback_pages(value, 0))
    assert observations(pages, decision_time(value, 0)) == ()
    assert StressFeedbackPage.from_json(pages[0].to_json()) == pages[0]
    later = observations(pages, decision_time(value, 1))
    if mode == "delayed":
        assert later == ()
        assert observations(pages, decision_time(value, 3))
    elif mode == "missing":
        assert later
        assert all(r.measurement.status == "unknown" and r.measurement.value is None for r in later)
    else:
        assert all(r.measurement.status == "observed" for r in later)
        truth = {r.execution_name: r.verdict for r in generate_build(value, 0)}
        assert all(
            r.measurement.value == (1 - truth[r.entity_id] if mode == "corrected" else truth[r.entity_id])
            for r in later
        )
    if mode == "full":
        assert len(later) == len(generate_build(value, 0))
    else:
        assert pages[0].logical_observations < len(generate_build(value, 0))


def test_receipt_cutoff_and_future_correction_do_not_rewrite_earlier_views():
    value = case(feedback="corrected", budget="full", variants="changing")
    pages = tuple(feedback_pages(value, 0))
    roots = [r for r in pages[0].history.revisions if r.supersedes is None]
    source_cutoff = roots[0].metadata.available_at
    assert observations(pages, source_cutoff) == ()
    before = visible_feedback(pages, decision_time(value, 1))[0]
    after = visible_feedback(pages, decision_time(value, 4))[0]
    assert all(
        a.measurement.value == 1 - b.measurement.value
        for a, b in zip(before.observations, after.observations, strict=True)
    )
    # Mutation deletes future corrections. Selected past fingerprint must stay identical.
    original = pages[0].model_dump()
    original["history"]["revisions"] = roots
    trimmed = StressFeedbackPage.model_validate(original)
    assert visible_feedback((trimmed,), decision_time(value, 1))[0].fingerprint == before.fingerprint
    # A fake early-label consumer would differ from the actual cutoff view.
    assert before.fingerprint != after.fingerprint
    with pytest.raises(ValueError, match="unavailable at its cutoff"):
        ObservationSnapshot.model_validate({**before.model_dump(), "observations": after.observations})
    assert observations(pages, source_cutoff + timedelta(minutes=1))


def test_large_corrected_feedback_pages_preserve_complete_lineages_and_bounds():
    value = case(size="large", feedback="corrected", budget="full", variants="changing", builds=2)
    pages = tuple(feedback_pages(value, 0))
    assert len(pages) == 25
    assert sum(len(p.history.revisions) for p in pages) == 100000
    assert all(len(p.history.revisions) <= 4096 for p in pages)
    assert len(observations(tuple(reversed(pages)), decision_time(value, 4))) == 50000
    for invalid in [pages[:-1], pages + pages[:1], ()]:
        with pytest.raises(ValueError, match="every distinct page"):
            visible_feedback(invalid, decision_time(value, 4))

    with pytest.raises(ValueError, match="74-page"):
        visible_feedback(pages * 3, decision_time(value, 4))
    changed = StressFeedbackPage.model_validate({**pages[0].model_dump(), "membership_digest": "sha256:" + "a" * 64})
    with pytest.raises(ValueError, match="conflicting bindings"):
        visible_feedback((changed,) + pages[1:], decision_time(value, 4))
    reordered = StressFeedbackPage.model_validate({**pages[0].model_dump(), "history": pages[1].history})
    counterpart = StressFeedbackPage.model_validate({**pages[1].model_dump(), "history": pages[0].history})
    with pytest.raises(ValueError, match="membership"):
        visible_feedback((reordered, counterpart) + pages[2:], decision_time(value, 4))


def test_page_membership_binding_and_identity_mutations_reject():
    value = case(feedback="full")
    pages = tuple(feedback_pages(value, 0))
    original = pages[0].model_dump()
    for change in [{"page_count": 2}, {"page_index": 1}, {"logical_observations": 0}]:
        with pytest.raises(ValueError, match="dimensions"):
            StressFeedbackPage.model_validate({**original, **change})
    bad = {**original, "case_fingerprint": "sha256:" + "a" * 64}
    with pytest.raises(ValueError, match="source/case"):
        StressFeedbackPage.model_validate(bad)
    future = pages[0].model_dump()
    future["history"]["source_version"] = "owned-ci-stress-feedback-v2"
    with pytest.raises(ValueError, match="source/case"):
        StressFeedbackPage.model_validate(future)
    for field, invalid, error in [
        ("observation_id", "wrong", "logical identity"),
        ("group_id", "other-group", "dependence group"),
    ]:
        changed = pages[0].model_dump()
        changed["history"]["revisions"][0]["metadata"][field] = invalid
        with pytest.raises(ValueError, match=error):
            StressFeedbackPage.model_validate(changed)
    bad = StressFeedbackPage.model_validate({**original, "membership_digest": "sha256:" + "b" * 64})
    with pytest.raises(ValueError, match="membership"):
        visible_feedback((bad,), decision_time(value, 1))


@pytest.mark.parametrize("bad", [-1, 33, True, 0.0])
def test_constructed_clock_bounds(bad):
    value = case()
    with pytest.raises(ValueError, match="decision index"):
        decision_time(value, bad)


def test_empty_selected_feedback_retains_one_empty_complete_page_and_no_invented_pass():
    value = case(variants="changing", budget="severe")
    ranking = tuple(reversed(tuple(row.execution_name for row in generate_build(value, 0))))
    pages = tuple(feedback_pages(value, 0, ranking))
    assert len(pages) == 1
    assert pages[0].logical_observations == 0
    assert pages[0].history.revisions == ()
    assert observations(pages, decision_time(value, 4)) == ()


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "foreign-measure"),
        ("unit", "s"),
        ("meaning", "priority-score-v1"),
        ("direction", "maximize"),
        ("minimum", None),
        ("value", 42.0),
        ("value", 0.5),
        ("value", -1.0),
    ],
)
def test_generic_measurements_cannot_impersonate_binary_feedback(field, value):
    page = next(feedback_pages(case(feedback="full"), 0)).model_dump()
    page["history"]["revisions"][0]["measurement"][field] = value
    with pytest.raises(ValueError):
        StressFeedbackPage.model_validate(page)


@pytest.mark.parametrize("status", ["unavailable", "not_applicable"])
def test_other_generic_null_states_do_not_impersonate_this_unknown_report_channel(status):
    page = next(feedback_pages(case(feedback="full"), 0)).model_dump()
    page["history"]["revisions"][0]["measurement"].update(value=None, status=status)
    with pytest.raises(ValueError, match="observed binary or unknown/null"):
        StressFeedbackPage.model_validate(page)


@pytest.mark.parametrize("mode", ["full", "censored", "delayed", "missing", "corrected"])
def test_reports_follow_actual_modeled_completion_and_immediate_next_decision_bound(mode):
    value = case(size="medium", duration="heavy_tail", feedback=mode, budget="full", workers="bounded")
    ends = {e.name: e.completed_seconds for e in schedule_build(value, 0).events}
    pages = tuple(feedback_pages(value, 0))
    interval = decision_time(value, 1) - decision_time(value, 0)
    for page in pages:
        assert page.observation_scope == "selected-modeled-execution"
        for revision in page.history.revisions:
            meta = revision.metadata
            assert meta.target_end is not None
            assert meta.target_end == meta.event_at + timedelta(seconds=ends[revision.entity_id])
            delay = 3 * interval if revision.supersedes else 2 * interval if mode == "delayed" else timedelta(0)
            assert meta.available_at == meta.target_end + timedelta(hours=1) + delay
            if not revision.supersedes and mode != "delayed":
                assert revision.received_at < decision_time(value, 1)


def test_large_serial_feedback_cannot_arrive_before_execution_finishes():
    value = case(
        size="large", duration="heavy_tail", failures="rare", variants="sparse", feedback="full", budget="full"
    )
    events = {e.name: e.completed_seconds for e in schedule_build(value, 0).events}
    assert max(events.values()) > 86400
    pages = tuple(feedback_pages(value, 0))
    assert all(
        r.metadata.available_at >= decision_time(value, 0) + timedelta(seconds=events[r.entity_id], hours=1)
        for p in pages
        for r in p.history.revisions
    )
    assert len(observations(pages, decision_time(value, 1))) == 50000


def test_full_feedback_at_severe_budget_is_a_declared_counterfactual():
    value = case(feedback="full", budget="severe")
    pages = tuple(feedback_pages(value, 0))
    assert all(p.observation_scope == "counterfactual-full-suite" for p in pages)
    assert len(observations(pages, decision_time(value, 1))) == len(generate_build(value, 0))
    with pytest.raises(ValueError, match="Full-label"):
        value.require_native()


def test_clock_uses_declared_bounds_not_future_labels_and_rejects_early_report_mutation(monkeypatch):
    import coleman.corpus.stress_world as world

    value = case(duration="failure_correlated", budget="full")
    before = decision_time(value, 1)
    with monkeypatch.context() as mutation:
        mutation.setattr(world, "_verdict", lambda *_: 1)
        assert decision_time(value, 1) == before
    assert decision_time(case(seed=29, duration="failure_correlated", budget="full"), 1) == before
    page = next(feedback_pages(value, 0)).model_dump()
    revision = page["history"]["revisions"][0]
    revision["metadata"]["available_at"] = revision["metadata"]["target_end"]
    revision["received_at"] = revision["metadata"]["available_at"] + timedelta(minutes=1)
    with pytest.raises(ValueError, match="follow modeled completion"):
        StressFeedbackPage.model_validate(page)


def test_missing_completion_and_changed_first_receipt_cannot_claim_generated_clock():
    value = case(feedback="full")
    for mutation in ["completion", "receipt"]:
        page = next(feedback_pages(value, 0)).model_dump()
        revision = page["history"]["revisions"][0]
        if mutation == "completion":
            revision["metadata"]["target_end"] = None
        else:
            revision["received_at"] += timedelta(seconds=1)
        with pytest.raises(ValueError, match="modeled completion|first receipt"):
            StressFeedbackPage.model_validate(page)


@pytest.mark.parametrize("duration", ["constant", "heavy_tail", "failure_correlated", "changing"])
@pytest.mark.parametrize("workers", ["serial", "bounded"])
def test_every_immediate_report_fits_before_next_decision_including_minute_receipt(duration, workers):
    value = case(duration=duration, workers=workers, feedback="full", budget="full")
    for page in feedback_pages(value, 0):
        assert all(r.received_at < decision_time(value, 1) for r in page.history.revisions)
