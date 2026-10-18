"""Deterministic human evidence reports; presentation never grants source admission."""

from __future__ import annotations

from html import escape

from coleman.corpus.admission_contracts import CorpusAdmission, ObservationProfile, SourceAssessment
from coleman.corpus.admission_sources import source_description


def _cell(value: str) -> str:
    """Escape source-derived Markdown/HTML, retaining one physical table row per value."""
    return escape(value, quote=True).replace("|", "\\|").replace("`", "&#96;").replace("\n", " ").replace("\r", " ")


def _overview(report: CorpusAdmission) -> list[str]:
    """Render a complete stable UTF-8 Markdown view from validated source facts.

    Reports preserve blocked/unassessed sources, original units, unknowns and
    narrow functional/synthetic purposes. Source/model/group counts are not
    independent projects or trials. Values are escaped as data. This performs
    no file/network access, repairs, subject execution or policy evaluation.
    Reading/presentation validates contracts; publication must verify sources.
    """
    lines = [
        "# CI corpus source admission and production reality",
        "",
        "Generalization: **insufficient evidence**. Source integrity, scoped qualification and protected confirmation",
        "remain separate. Negative/inconclusive observations are retained. Seeds, variants and trials are dependent.",
        "",
        "Machine evidence: [admission.json](admission.json). No policy superiority or savings are established.",
        "",
        "## Required diversity",
        "",
        "| Regime | Source-purpose coverage |",
        "| --- | --- |",
    ]
    for regime in report.registry.required_regimes:
        status = "covered for the stated source purpose" if regime in report.covered_regimes else "uncovered"
        lines.append(f"| {_cell(regime)} | {status} |")
    models = ", ".join(report.observed_source_models) or "unavailable"
    lines += [
        "",
        "Reviewed observed source structures: " + _cell(models) + ".",
        "Observed source-model count: " + str(len(report.observed_source_models)) + "; at least two are required.",
        "Source structures retain job/stage identities; this does not clear missing temporal or operational evidence.",
        "",
        "## Sources",
        "",
        "| Project/family | Scope | History | Retained rows | Source groups | Logical source identity |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    return lines


def _measurements(profile: ObservationProfile) -> list[str]:
    """Render scalar observations without replacing unavailable values with zero."""
    lines = []
    for measure in profile.measurements:
        value = "null" if measure.value is None else str(measure.value)
        values = (measure.name, value, measure.unit, measure.status, measure.meaning)
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    return lines


def _duration(profile: ObservationProfile) -> list[str]:
    """Render signed/nonfinite/null/zero categories and explicitly descriptive quantiles."""
    lines = []
    duration = profile.duration
    lines += [
        "",
        "Duration distribution: " + _cell(duration.semantics.unit) + "; " + _cell(duration.semantics.meaning) + ".",
        "Finite/null/nonfinite/negative/zero counts: "
        + "/".join(
            str(value)
            for value in (
                duration.finite_count,
                duration.null_count,
                duration.nonfinite_count,
                duration.negative_count,
                duration.zero_count,
            )
        )
        + ".",
        "Minimum/median/p95/maximum: "
        + "/".join(
            "null" if value is None else str(value)
            for value in (
                duration.minimum,
                duration.median,
                duration.p95,
                duration.maximum,
            )
        )
        + ".",
        "Exact reported decimals/harness ticks remain in source receipts; binary64 distributions are descriptive.",
        "",
        "### Production reality",
        "",
        "| Axis | Status | Values | Evidence locator | Limitation |",
        "| --- | --- | --- | --- | --- |",
    ]
    return lines


def _reality(item: SourceAssessment) -> list[str]:
    """Escape all qualitative axes; declarations remain separate from physical runtime evidence."""
    lines = []
    for fact in item.reality:
        values = (
            fact.axis,
            fact.status,
            ", ".join(fact.values) if fact.values else "null",
            fact.locator,
            fact.limitation,
        )
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    return lines


def _source_details(item: SourceAssessment, index: int) -> list[str]:
    """Link one source's purpose, measurements and limitations to its machine-readable evidence."""
    lines = [
        "",
        "## Source " + str(index + 1) + ": " + _cell(source_description(item.payload).project_group),
        "",
        "Logical identity: " + _cell(item.card.manifest.benchmark_id) + ".",
        "Declared source grouping: " + _cell(", ".join(item.profile.identity.group_fields)) + ".",
        "Counted execution-group dimensions: " + _cell(", ".join(item.profile.execution_group_fields)) + ".",
        "Source/payload/review/card: `assessments[" + str(index) + "]` in the machine report.",
        "Primary terms apply only to the primary source; all embedded component/input terms remain distinct.",
        "Historical blockers: " + _cell(", ".join(item.historical_blockers)) + ".",
        "Scoped qualification blockers: "
        + _cell(", ".join(item.admission_reasons) or "none for this narrow purpose")
        + ".",
        "",
        "### Descriptive measurements",
        "",
        "| Measurement | Value | Unit | Status | Meaning |",
        "| --- | ---: | --- | --- | --- |",
    ]
    lines += _measurements(item.profile)
    lines += _duration(item.profile)
    lines += _reality(item)
    lines += ["", "Source-context review limitations:", ""]
    lines.extend("- " + _cell(value) for value in item.review.limitations)
    return lines


def _source_rows(report: CorpusAdmission) -> list[str]:
    """Present group counts without calling them independent project/trial counts."""
    lines = []
    for item in report.assessments:
        description = source_description(item.payload)
        values = (
            description.project_group,
            item.scope,
            item.card.history,
            str(item.profile.rows),
            str(item.profile.source_groups),
            item.card.manifest.benchmark_id,
        )
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    return lines


def _unbound(report: CorpusAdmission) -> list[str]:
    """Retain every unbound registration and its original blockers."""
    lines = []
    lines += [
        "",
        "Groups preserve source dimensions and shared project/fault families; they are not independent sampling units.",
        "No genuinely mapped unique-fault count is available. Curated fault labels are not a unique-fault mapping.",
        "",
        "## Registrations without this audit's source binding",
        "",
    ]
    for key in report.unassessed_sources:
        card = report.registry.card(key)
        reasons = card.manifest.blocking_reasons or ("source_not_bound_to_this_audit",)
        lines.append("- " + _cell(key) + ": " + _cell(", ".join(reasons)))
    if not report.unassessed_sources:
        lines.append(
            "None. Binding every source still does not establish complete diversity or protected confirmation."
        )
    return lines


def render_corpus_admission(report: CorpusAdmission) -> str:
    """Render stable escaped Markdown from validated source-purpose evidence.

    Preserves blocked/unassessed sources, signed/zero/null distinctions and
    narrow functional/synthetic scopes. Source groups are not independent
    sampling units. Performs no I/O or execution; publication separately verifies
    actual bytes. Unknown versions and inconsistent evidence raise ValueError.
    """
    report = CorpusAdmission.model_validate(report)
    lines = _overview(report) + _source_rows(report) + _unbound(report)
    for index, item in enumerate(report.assessments):
        lines += _source_details(item, index)
    return "\n".join(lines) + "\n"
