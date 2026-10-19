#!/usr/bin/env python3
"""Audit NumPy parameter documentation without importing or executing inspected code.

The audit covers explicitly defined public functions/methods and constructors in
public classes. Nested/private functions are excluded; constructor parameters may
be documented on the class. This checks signature coverage, not semantic accuracy,
return/exception completeness or the stability of a public API.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from numpydoc.docscrape import NumpyDocString


def _documented_parameters(docstring: str | None) -> set[str]:
    """Parse exact NumPy Parameters entries, including grouped and starred names."""
    parsed = NumpyDocString(docstring or "")
    return {name.strip().lstrip("*") for parameter in parsed["Parameters"] for name in parameter.name.split(",")}


def _functions(nodes: Iterable[ast.stmt], prefix: str = "", class_doc: str | None = None):
    """Yield reachable public definitions, excluding nested function-local helpers."""
    for node in nodes:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            yield from _functions(node.body, f"{prefix}{node.name}.", ast.get_docstring(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            not node.name.startswith("_") or (prefix and node.name == "__init__")
        ):
            yield f"{prefix}{node.name}", node, class_doc


def _python_files(paths: Iterable[Path]) -> list[Path]:
    """Expand caller-owned local directories and deduplicate files in stable order."""
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(path.rglob("*.py"))
        else:
            files.add(path)
    return sorted(files)


def _parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef, *, is_method: bool) -> list[str]:
    """Keep explicit parameters while removing only an implicit method receiver."""
    positional = [*node.args.posonlyargs, *node.args.args]
    arguments = [arg.arg for arg in (*positional, *node.args.kwonlyargs)]
    static = any(isinstance(item, ast.Name) and item.id == "staticmethod" for item in node.decorator_list)
    if is_method and positional and not static:
        arguments.remove(positional[0].arg)
    arguments.extend(arg.arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None)
    return arguments


def audit_paths(paths: Iterable[Path]) -> dict[str, Any]:
    """Inspect local signatures and NumPy parameter entries without executing code.

    Parameters
    ----------
    paths : collections.abc.Iterable[pathlib.Path]
        Trusted local Python files or directories to inspect recursively.
        Repeated paths are deduplicated.

    Returns
    -------
    dict[str, Any]
        Versioned audit containing file/signature counts and ordered findings.

    Raises
    ------
    OSError
        If an input cannot be read.
    SyntaxError
        If a source file cannot be parsed as Python.
    ValueError
        If the NumPy documentation parser rejects a docstring.

    Notes
    -----
    Reads source text without importing inspected modules or writing files.
    Callers own the local path selection and resource limits. This reports
    parameter-name coverage, not semantic or full docstring correctness.
    """
    files = _python_files(paths)
    findings = []
    checked = 0
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name, node, class_doc in _functions(tree.body):
            arguments = _parameter_names(node, is_method="." in name)
            if not arguments:
                continue
            checked += 1
            documented = _documented_parameters(ast.get_docstring(node))
            if node.name == "__init__":
                documented |= _documented_parameters(class_doc)
            missing = sorted(set(arguments) - documented)
            if missing:
                findings.append({"path": str(path), "line": node.lineno, "name": name, "missing_parameters": missing})
    return {"schema_version": 1, "files": len(files), "signatures_checked": checked, "findings": findings}


def main() -> int:
    """Print a stable local audit with an optional parameter-coverage gate.

    Returns
    -------
    int
        One for parameter findings with ``--check``, otherwise zero.

    Notes
    -----
    Reads arguments and local sources and writes JSON to standard output.
    Parsing and I/O errors propagate; inspected modules are never executed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("coleman")])
    parser.add_argument("--check", action="store_true", help="Return a nonzero status when parameters are undocumented")
    args = parser.parse_args()
    report = audit_paths(args.paths)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(args.check and bool(report["findings"]))


if __name__ == "__main__":
    raise SystemExit(main())
