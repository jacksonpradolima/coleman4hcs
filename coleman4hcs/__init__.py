"""Backward-compatible alias for the ``coleman`` package.

This package re-exports everything from ``coleman`` so that existing code
importing ``coleman4hcs`` (including deep sub-modules such as
``coleman4hcs.policy.base``) continues to work without changes.

A :class:`MetaPathFinder` is installed at import time that redirects every
``coleman4hcs.*`` import to the corresponding ``coleman.*`` module.  The
redirected module object is identical to the canonical one (same identity),
so singletons such as the policy RNG are *not* duplicated.
"""

from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec


class _AliasLoader(Loader):
    """Return the canonical ``coleman.*`` module when asked to load a ``coleman4hcs.*`` one."""

    def __init__(self, canonical_name: str) -> None:
        self._canonical = canonical_name

    def create_module(self, spec: ModuleSpec):  # noqa: ARG002
        # Return the already-imported (or freshly imported) canonical module.
        # Checking sys.modules first avoids re-executing the module body.
        mod = sys.modules.get(self._canonical)
        if mod is None:
            mod = importlib.import_module(self._canonical)
        return mod

    def exec_module(self, module) -> None:  # noqa: ARG002
        # The module was fully populated by create_module; nothing to do here.
        pass


class _Coleman4HCSFinder(MetaPathFinder):
    """Redirect every ``coleman4hcs.*`` import to its ``coleman.*`` equivalent."""

    _PREFIX = "coleman4hcs."
    _CANONICAL_PREFIX = "coleman."

    def find_spec(self, fullname: str, path, target=None):  # noqa: ARG002
        if not fullname.startswith(self._PREFIX):
            return None
        canonical = self._CANONICAL_PREFIX + fullname[len(self._PREFIX):]
        try:
            importlib.import_module(canonical)
        except ImportError:
            return None
        return ModuleSpec(fullname, _AliasLoader(canonical))


# Install the finder at position 0 so it takes priority over the default
# file-system finders (which would otherwise fail to find coleman4hcs sub-modules).
sys.meta_path.insert(0, _Coleman4HCSFinder())
