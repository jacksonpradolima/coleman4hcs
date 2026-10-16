"""Optional public HTTPS source adapter; no redirects, proxies, credentials or implicit core imports."""

from __future__ import annotations

import http.client
import re
import ssl
from collections.abc import Iterator
from contextlib import contextmanager
from time import monotonic
from urllib.parse import urlsplit

from coleman.corpus.materialization_contracts import ByteReader

# Provider expansion requires an explicitly reviewed public-source integration.
_PUBLIC_HOSTS = frozenset({"raw.githubusercontent.com", "repo.maven.apache.org", "zenodo.org"})


def _public_target(url: str) -> tuple[str, str]:
    """Accept only a reviewed public HTTPS origin and uncredentialed, query-free canonical URL."""
    if any(character.isspace() or ord(character) < 32 for character in url):
        raise ValueError("Source URL contains whitespace or control characters")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _PUBLIC_HOSTS
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise ValueError("Source URL must use an approved public HTTPS origin without credentials/query/fragment")
    return parsed.hostname, parsed.path


def _identity_response(response: http.client.HTTPResponse, size: int) -> None:
    """Require one complete known-size entity without redirects, transfer/content compression or partial content."""
    if response.status != 200:
        raise ValueError("Public source must return HTTP 200 without redirects or partial content")
    if response.headers.get_all("Transfer-Encoding") or response.headers.get_all("Content-Encoding", []) not in (
        [],
        ["identity"],
    ):
        raise ValueError("Public source must provide an unencoded known-size entity")
    lengths = response.headers.get_all("Content-Length", [])
    if len(lengths) != 1 or re.fullmatch(r"\d+", lengths[0], flags=re.ASCII) is None or int(lengths[0]) != size:
        raise ValueError("Public source Content-Length disagrees with frozen byte count")


class _DeadlineReader:
    """Check elapsed budget between reads; socket inactivity timeout bounds individual blocking I/O."""

    def __init__(self, response: http.client.HTTPResponse, deadline: float) -> None:
        """Retain caller-owned response and monotonic deadline, without reading source bytes."""
        self.response = response
        self.deadline = deadline

    def read(self, maximum: int, /) -> bytes:
        """Read a bounded entity chunk; timeout raises before/after I/O and leaves no completed cache blob."""
        if monotonic() >= self.deadline:
            raise TimeoutError("Public source exceeded elapsed acquisition budget")
        chunk = self.response.read(maximum)
        if monotonic() >= self.deadline:
            raise TimeoutError("Public source exceeded elapsed acquisition budget")
        return chunk


@contextmanager
def public_https_source(
    url: str, size_bytes: int, *, socket_timeout: float = 15.0, elapsed_budget: float = 60.0
) -> Iterator[ByteReader]:
    """Open a reviewed public source with certificate verification and no ambient authentication/proxy state.

    The adapter never follows redirects or requests private/authenticated sources.
    Only raw.githubusercontent.com, repo.maven.apache.org and zenodo.org are supported. The caller separately
    supplies a licensed, digest-pinned materialization spec. Size is the HTTP
    entity byte count, not packet/TLS/header overhead. Socket inactivity timeouts
    and elapsed checks between reads are not a hard OS deadline for DNS or headers.
    HTTP/SSL/I/O failures propagate; connection/response close on every exit.
    """
    host, path = _public_target(url)
    if type(size_bytes) is not int or not 0 <= size_bytes <= 8 * 1024**3:
        raise ValueError("Public source requires a bounded integer entity byte count")
    if (
        type(socket_timeout) not in {int, float}
        or type(elapsed_budget) not in {int, float}
        or not 0 < socket_timeout <= 30
        or not socket_timeout <= elapsed_budget <= 300
    ):
        raise ValueError("Public source requires bounded socket and elapsed budgets")
    deadline = monotonic() + elapsed_budget
    context = ssl.create_default_context()
    context.set_alpn_protocols(["http/1.1"])
    connection = http.client.HTTPSConnection(host, timeout=socket_timeout, context=context)
    try:
        connection.request(
            "GET", path, headers={"Accept-Encoding": "identity", "User-Agent": "COLEMAN/corpus-integrity-v1"}
        )
        with connection.getresponse() as response:
            _identity_response(response, size_bytes)
            yield _DeadlineReader(response, deadline)
    except http.client.HTTPException as exc:
        raise OSError("Public HTTPS source protocol failure") from exc
    finally:
        connection.close()
