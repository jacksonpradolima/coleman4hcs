# Source materialization API

Use `SourceCache` and a frozen `MaterializationSpec` to verify permitted source
bytes and selected archive members. Network readers are explicit adapters;
validation works offline. Materialization does not grant scientific admission.

Start with [the source-cache workflow](../corpus-materialization.md) or choose a
[dataset adapter](../datasets.md). The docstrings below describe byte bounds,
side effects, errors and concurrency limits.

::: coleman.corpus.materialization
