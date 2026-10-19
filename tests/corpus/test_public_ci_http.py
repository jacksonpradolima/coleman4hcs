"""Exercise the real anonymous transport's bounded protocol handling with owned in-memory HTTP responses."""

from email.message import Message
from io import BytesIO

import pytest

from tests.corpus.public_actions_support import example_module


class OwnedResponse(BytesIO):
    """Small owned HTTP boundary with realistic headers and streaming reads."""

    status = 200

    def __init__(self, payload, *, content_type="application/json", selected="2022-11-28"):
        super().__init__(payload)
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.headers["X-GitHub-Api-Version-Selected"] = selected


def setup_http(monkeypatch, payload=b'{"owned":true}', **kwargs):
    """Replace only the external socket; the real fetcher creates headers, bounds bytes and validates contracts."""
    module = example_module()
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            assert timeout == 10
            return OwnedResponse(payload, **kwargs)

    monkeypatch.setattr(module.urllib.request, "build_opener", lambda handler: Opener())
    return module, requests


def test_anonymous_fixed_version_transport_records_original_bytes(monkeypatch):
    module, requests = setup_http(monkeypatch)
    value = module.fetch_response(module.API + "/repos/coleman-owned/ci-example/actions/runs/42", "run")
    assert value.body == '{"owned":true}'
    assert value.selected_api_version == module.API_VERSION
    headers = dict(requests[0].header_items())
    assert headers["X-github-api-version"] == "2022-11-28"
    assert "Authorization" not in headers
    assert "Cookie" not in headers
    assert module.Response.from_json(value.to_json()) == value


@pytest.mark.parametrize(
    ("payload", "kwargs"),
    [
        (b"x" * (2 * 1024**2 + 1), {}),
        (b'{"a":1,"a":2}', {}),
        (b'{"a":NaN}', {}),
        (b"\xff", {}),
        (b"{}", {"content_type": "text/html"}),
        (b"{}", {"selected": "2099-01-01"}),
    ],
)
def test_real_transport_rejects_unbounded_or_unsupported_response(monkeypatch, payload, kwargs):
    module, _ = setup_http(monkeypatch, payload, **kwargs)
    with pytest.raises(ValueError):
        module.fetch_response(module.API + "/repos/coleman-owned/ci-example/actions/runs/42", "run")


def test_foreign_origin_and_redirect_are_refused(monkeypatch):
    module, requests = setup_http(monkeypatch)
    with pytest.raises(ValueError, match="fixed public API origin"):
        module.fetch_response("https://example.invalid/repos/coleman-owned/ci-example", "run")
    assert requests == []
    assert module._NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://example.invalid") is None
