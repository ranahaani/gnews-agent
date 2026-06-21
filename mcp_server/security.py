"""Webhook URL validator — SSRF defense for ``monitor_topic``.

When the MCP server is exposed (especially over HTTP), webhook URLs flow in
from untrusted MCP clients. Letting them point at RFC1918, loopback,
link-local, or cloud-metadata addresses turns the server into an SSRF
pivot. This validator refuses those.

Local-Python-API callers (``NewsMemory.monitor(...)`` invoked in-process)
can opt into ``allow_http=True`` since the threat model there is different.
The MCP layer always passes ``allow_http=False``.
"""
from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from gnews_agent.exceptions import WebhookSecurityError


# Cloud metadata + AWS link-local + GCP/Azure metadata.
_METADATA_HOSTS = frozenset({
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.azure.com",
})

_ALLOWED_SCHEMES = frozenset({"http", "https"})


def validate_webhook(url: str, *, allow_http: bool = False) -> str:
    """Return the URL unchanged if it passes; raise :class:`WebhookSecurityError` otherwise.

    Checks
    ------
    * scheme must be http or https
    * https required unless ``allow_http=True`` (set by local API callers only)
    * host present
    * cloud-metadata hostnames blocked outright
    * IPv4/IPv6 hosts: refuse private, loopback, link-local, multicast,
      reserved, and unspecified ranges
    """
    if not url:
        raise WebhookSecurityError("webhook URL is empty")
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise WebhookSecurityError(f"webhook scheme {scheme!r} not in {sorted(_ALLOWED_SCHEMES)}")
    if scheme == "http" and not allow_http:
        raise WebhookSecurityError("https required for webhooks (set allow_http=True for local API)")
    host = parts.hostname
    if not host:
        raise WebhookSecurityError(f"webhook host missing in {url!r}")
    host_lower = host.lower()
    if host_lower in _METADATA_HOSTS:
        raise WebhookSecurityError(f"webhook host {host_lower!r} is a cloud-metadata endpoint")
    try:
        addr = ipaddress.ip_address(host_lower)
    except ValueError:
        # Hostname rather than literal IP — DNS resolution time-of-check vs
        # time-of-use is a known gap; the network egress layer (or a future
        # outbound proxy) must enforce. For v1 we refuse only the literal-IP
        # case where the SSRF intent is unambiguous.
        return url
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    ):
        raise WebhookSecurityError(f"webhook IP {addr} is in a blocked range")
    return url
