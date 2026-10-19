# Artifact publication API

Use an immutable writer for typed content-addressed evidence. The legacy atomic
JSON/CSV helpers replace mutable single-writer exports. Their contracts differ
in concurrency, identity, validation and interruption behavior.

Read [architecture](../setup.md#identity-and-reproducibility) and
[versioned results](../domain-results.md) for the relationships to native result
sinks and checkpoints. Callers own trusted local paths and parent directories.

::: coleman.artifacts
