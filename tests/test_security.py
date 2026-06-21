"""Webhook SSRF validator — must block private/loopback/link-local/metadata."""
from __future__ import annotations

import pytest

from gnews_agent.exceptions import WebhookSecurityError
from mcp_server.security import validate_webhook


class TestAcceptedUrls:
    def test_public_https(self):
        assert validate_webhook("https://hooks.example.com/notify") == "https://hooks.example.com/notify"

    def test_hostname_unresolved_allowed(self):
        # DNS-based SSRF is out of scope for the v1 literal-IP guard.
        assert validate_webhook("https://internal-but-public-dns.example.com/x")

    def test_http_allowed_when_local(self):
        assert validate_webhook("http://hooks.example.com/notify", allow_http=True)


class TestRejectedUrls:
    @pytest.mark.parametrize("url", [
        "",
        "ftp://example.com/x",
        "javascript:alert(1)",
        "file:///etc/passwd",
        "https:///no-host",
    ])
    def test_bad_scheme_or_host(self, url):
        with pytest.raises(WebhookSecurityError):
            validate_webhook(url)

    def test_http_rejected_by_default(self):
        with pytest.raises(WebhookSecurityError):
            validate_webhook("http://hooks.example.com/x")

    @pytest.mark.parametrize("url", [
        "https://127.0.0.1/x",          # loopback
        "https://10.0.0.1/x",           # RFC1918
        "https://192.168.1.1/x",        # RFC1918
        "https://172.16.5.5/x",         # RFC1918
        "https://169.254.0.5/x",        # link-local
        "https://169.254.169.254/x",    # AWS/GCP metadata literal
        "https://[::1]/x",              # IPv6 loopback
        "https://[fe80::1]/x",          # IPv6 link-local
        "https://[fc00::1]/x",          # IPv6 ULA
    ])
    def test_private_or_link_local_ip_rejected(self, url):
        with pytest.raises(WebhookSecurityError):
            validate_webhook(url)

    def test_metadata_hostname_rejected(self):
        with pytest.raises(WebhookSecurityError):
            validate_webhook("https://metadata.google.internal/computeMetadata/v1/")
