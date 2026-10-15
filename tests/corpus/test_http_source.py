"""Optional HTTP transport protocol tests, using no network and no mocked materialization kernel."""

from email.message import Message
from unittest.mock import Mock

import pytest

from coleman.transports import corpus_http


@pytest.mark.parametrize(
    "url",
    [
        "http://raw.githubusercontent.com/a",
        "https://localhost/a",
        "https://127.0.0.1/a",
        "https://raw.githubusercontent.com:444/a",
        "https://user:secret@raw.githubusercontent.com/a",
        "https://raw.githubusercontent.com/a?token=x",
        "https://raw.githubusercontent.com/a#fragment",
        "https://raw.githubusercontent.com/a\nb",
        "file:///tmp/a",
        "https://example.org/a",
    ],
)
def test_unapproved_urls_reject_before_connection(monkeypatch, url):
    connect = Mock()
    monkeypatch.setattr(corpus_http.http.client, "HTTPSConnection", connect)
    with pytest.raises(ValueError), corpus_http.public_https_source(url, 3):
        pytest.fail("Unapproved source opened")
    connect.assert_not_called()


def response(headers=None, status=200, payload=b"abc"):
    result = Mock()
    result.status = status
    result.headers = Message()
    for key, value in headers or [("Content-Length", "3")]:
        result.headers[key] = value
    result.read.return_value = payload
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result


def connection(monkeypatch, reply):
    client = Mock()
    client.getresponse.return_value = reply
    constructor = Mock(return_value=client)
    monkeypatch.setattr(corpus_http.http.client, "HTTPSConnection", constructor)
    return client, constructor


def test_approved_origin_uses_verified_tls_no_ambient_headers_and_closes(monkeypatch):
    reply = response()
    client, constructor = connection(monkeypatch, reply)
    with corpus_http.public_https_source("https://raw.githubusercontent.com/owner/ci/pinned/data", 3) as source:
        assert source.read(3) == b"abc"
    args, kwargs = constructor.call_args
    assert args == ("raw.githubusercontent.com",)
    assert kwargs["context"].check_hostname
    assert kwargs["context"].verify_mode == corpus_http.ssl.CERT_REQUIRED
    request = client.request.call_args
    assert request.args == ("GET", "/owner/ci/pinned/data")
    assert set(request.kwargs["headers"]) == {"Accept-Encoding", "User-Agent"}
    client.close.assert_called_once()
    reply.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "headers,status",
    [
        ([("Content-Length", "3")], 302),
        ([("Content-Length", "3")], 206),
        ([("Content-Length", "2")], 200),
        ([("Content-Length", "x")], 200),
        ([("Content-Length", "3"), ("Content-Length", "3")], 200),
        ([("Content-Length", "3"), ("Transfer-Encoding", "chunked")], 200),
        ([("Content-Length", "3"), ("Content-Encoding", "gzip")], 200),
        ([("Content-Length", "3"), ("Content-Encoding", "identity"), ("Content-Encoding", "gzip")], 200),
        ([("Unrelated", "header")], 200),
    ],
)
def test_response_status_length_and_encoding_reject_and_close(monkeypatch, headers, status):
    client, _ = connection(monkeypatch, response(headers, status))
    with pytest.raises(ValueError), corpus_http.public_https_source("https://raw.githubusercontent.com/a", 3):
        pytest.fail("Invalid source response opened")
    client.close.assert_called_once()


def test_elapsed_budget_and_protocol_errors_close(monkeypatch):
    reply = response()
    client, _ = connection(monkeypatch, reply)
    monkeypatch.setattr(corpus_http, "monotonic", Mock(side_effect=[0.0, 1.0, 61.0]))
    with (
        pytest.raises(TimeoutError),
        corpus_http.public_https_source("https://raw.githubusercontent.com/a", 3) as source,
    ):
        source.read(3)
    client.close.assert_called_once()
    monkeypatch.setattr(corpus_http, "monotonic", lambda: 0.0)
    client, _ = connection(monkeypatch, reply)
    client.getresponse.side_effect = corpus_http.http.client.BadStatusLine("invalid")
    with (
        pytest.raises(OSError, match="protocol"),
        corpus_http.public_https_source("https://raw.githubusercontent.com/a", 3),
    ):
        pytest.fail("Bad protocol opened")
    client.close.assert_called_once()


@pytest.mark.parametrize(
    "size,timeout,budget",
    [
        (True, 15.0, 60.0),
        (-1, 15.0, 60.0),
        (3, 0.0, 60.0),
        (3, 31.0, 60.0),
        (3, 15.0, 1.0),
        (3, True, 60.0),
        (3, 15.0, float("nan")),
        (3, 15.0, float("inf")),
    ],
)
def test_invalid_budget_and_size_parameters_make_no_connection(monkeypatch, size, timeout, budget):
    connect = Mock()
    monkeypatch.setattr(corpus_http.http.client, "HTTPSConnection", connect)
    with (
        pytest.raises(ValueError),
        corpus_http.public_https_source(
            "https://raw.githubusercontent.com/a", size, socket_timeout=timeout, elapsed_budget=budget
        ),
    ):
        pytest.fail("Invalid budget opened")
    connect.assert_not_called()


def test_expired_budget_rejects_before_response_body_read(monkeypatch):
    reply = response()
    client, _ = connection(monkeypatch, reply)
    monkeypatch.setattr(corpus_http, "monotonic", Mock(side_effect=[0.0, 61.0]))
    with (
        pytest.raises(TimeoutError),
        corpus_http.public_https_source("https://raw.githubusercontent.com/a", 3) as source,
    ):
        source.read(3)
    reply.read.assert_not_called()
    client.close.assert_called_once()
