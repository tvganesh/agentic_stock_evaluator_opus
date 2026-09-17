"""The egress gate: the only code in the system allowed to issue an HTTP request to Upstox.

Callers never supply a URL. They name a :class:`~sealed_window.governance.policy.CapabilityName`
and pass typed parameters; the gate builds the URL from the policy template and then
authorises the *finished* URL with an independent check (:func:`authorise`) that knows
nothing about how it was built. The same check runs again on the wire-level request via
an httpx event hook, so even a URL normalised by the HTTP library is re-verified.

Checks, all deny-by-default:

* method is ``GET``; scheme is ``https``; no userinfo, explicit port or fragment;
* host is an allowlisted Upstox host;
* path matches an allowlisted template exactly, segment by segment, with every variable
  segment fully matching its typed pattern (no ``..``, no empty segments, no encoded ``/``);
* query parameters are exactly the allowlisted names, no duplicates, typed values, and all
  required parameters present (this is what denies ``/v2/news?category=holdings``);
* the Authorization header is attached only for capabilities that need it, and a request
  to any other host carrying one is denied;
* redirects are never followed; bodies are streamed with a size cap; a per-run request
  cap bounds runaway loops; proxy/netrc environment settings are ignored;
* a rolling rate limiter keeps every request inside Upstox's per-second, per-minute and
  per-30-minute limits, and an HTTP 429 raises :class:`EgressRateLimited`, which the ETL
  treats as fatal (stop and resume later) rather than retrying into a suspension.

Every allow and deny is written to the audit log. This module is loaded only in the
ACQUIRE process role and is import-forbidden in the SEALED role.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

import httpx

from . import policy
from .audit import AuditLog
from .credentials import AnalyticsToken
from .errors import EgressDenied
from .ratelimit import RollingRateLimiter
from .seal import SEAL


class EgressHTTPError(RuntimeError):
    """A permitted request that returned a non-2xx status; the ETL records it and skips the row.

    Deliberately not a GovernanceViolation: the vendor failing is not a policy breach.
    """

    def __init__(self, capability: policy.CapabilityName, status_code: int) -> None:
        """Record which capability failed and with what HTTP status."""
        super().__init__(f"{capability.value} returned HTTP {status_code}")
        self.capability = capability
        self.status_code = status_code


class EgressRateLimited(EgressHTTPError):
    """Upstox answered HTTP 429. The acquisition must stop (and resume later), never keep hammering."""


@dataclass(frozen=True)
class AuthorisedRequest:
    """Proof that a (method, URL) pair passed :func:`authorise`, tied to the matching capability."""

    capability: policy.Capability
    url: str
    path_params: Mapping[str, str]
    query_params: Mapping[str, str]


def build_url(
    capability: policy.Capability,
    path_params: Mapping[str, str],
    query_params: Mapping[str, str],
) -> str:
    """Build a URL for ``capability`` from typed parameters, validating each one first.

    Values are checked against the policy rules *before* substitution and percent-encoded,
    so a parameter can never inject a path segment or extra query key.
    """
    expected = set(capability.path_params)
    if set(path_params) != expected:
        raise EgressDenied(
            f"{capability.name.value}: path params {sorted(path_params)} != allowed {sorted(expected)}"
        )
    for name, value in path_params.items():
        if not capability.path_params[name].matches(value):
            raise EgressDenied(f"{capability.name.value}: path param {name!r} has an invalid value")
    path = capability.path_template.format(**{k: quote(v, safe="") for k, v in path_params.items()})
    _check_query(capability, dict(query_params))
    query = urlencode(sorted(query_params.items()))
    return f"{policy.ALLOWED_SCHEME}://{capability.host}{path}" + (f"?{query}" if query else "")


def _check_query(capability: policy.Capability, query: Mapping[str, str]) -> None:
    """Validate query parameters against the capability: known names, typed values, required present."""
    for name, value in query.items():
        rule = capability.query_params.get(name)
        if rule is None:
            raise EgressDenied(f"{capability.name.value}: query param {name!r} is not allowlisted")
        if not rule.matches(value):
            raise EgressDenied(f"{capability.name.value}: query param {name!r} has a disallowed value")
    for name, rule in capability.query_params.items():
        if rule.required and name not in query:
            raise EgressDenied(f"{capability.name.value}: required query param {name!r} missing")


def _match_path(capability: policy.Capability, raw_path: str) -> dict[str, str] | None:
    """Match a raw (still percent-encoded) path against a capability template, segment by segment.

    Returns the decoded variable segments on a match, or ``None`` if this template does not apply.
    """
    template_parts = capability.path_template.split("/")
    path_parts = raw_path.split("/")
    if len(template_parts) != len(path_parts):
        return None
    params: dict[str, str] = {}
    for template_part, path_part in zip(template_parts, path_parts):
        if template_part.startswith("{") and template_part.endswith("}"):
            name = template_part[1:-1]
            value = unquote(path_part)
            if "/" in value or not capability.path_params[name].matches(value):
                return None
            params[name] = value
        elif unquote(path_part) != template_part:
            return None
    return params


def authorise(method: str, url: str) -> AuthorisedRequest:
    """Authorise a fully-formed request against the allowlist, or raise :class:`EgressDenied`.

    This is the reference check used by the gate, by the wire-level event hook and by the
    denied-path test suite. It is deliberately independent of :func:`build_url`.
    """
    if method.upper() != policy.ALLOWED_METHOD or method != method.upper():
        raise EgressDenied(f"method {method!r} is not permitted (GET only)")
    parts = urlsplit(url)
    if parts.scheme != policy.ALLOWED_SCHEME:
        raise EgressDenied(f"scheme {parts.scheme!r} is not permitted")
    host = parts.hostname or ""
    if parts.netloc != host:
        raise EgressDenied("userinfo, explicit ports and non-canonical hosts are not permitted")
    if host not in policy.UPSTOX_HOSTS:
        raise EgressDenied(f"host {host!r} is not allowlisted")
    if parts.fragment:
        raise EgressDenied("URL fragments are not permitted")
    segments = parts.path.split("/")[1:]
    if not parts.path.startswith("/") or any(
        seg == "" or unquote(seg) in (".", "..") for seg in segments
    ):
        raise EgressDenied("path must be absolute with no empty or dot segments")

    try:
        pairs = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=bool(parts.query))
    except ValueError as exc:
        raise EgressDenied("malformed query string") from exc
    names = [name for name, _ in pairs]
    if len(names) != len(set(names)):
        raise EgressDenied("duplicate query parameters are not permitted")
    query = dict(pairs)

    for capability in policy.CAPABILITIES.values():
        if capability.host != host:
            continue
        path_params = _match_path(capability, parts.path)
        if path_params is None:
            continue
        _check_query(capability, query)
        return AuthorisedRequest(capability, url, path_params, query)
    raise EgressDenied(f"no allowlisted capability matches {host}{parts.path}")


class EgressGate:
    """Holds the analytics token and the HTTP client; turns capability requests into bytes.

    The ETL adapter (``sealed_window.acquire.upstox_adapter``) is the only caller. Closing
    the gate drops the token, which is part of sealing the acquisition window.
    """

    def __init__(
        self,
        token: AnalyticsToken,
        audit: AuditLog,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        prior_request_ages: Iterable[float] = (),
    ) -> None:
        """Create the gate with a private HTTP client; ``transport`` lets tests stub the wire.

        ``prior_request_ages`` are seconds-ago timestamps of requests earlier runs sent, counted
        against the rate limits because Upstox limits the account, not the process.
        """
        self._token = token
        self._audit = audit
        self._clock = clock
        self._sleep = sleep
        self._last_request_at: float | None = None
        self._request_count = 0
        self._limiter = RollingRateLimiter(policy.UPSTOX_RATE_LIMITS, clock=clock, sleep=sleep)
        seeded = self._limiter.seed(prior_request_ages)
        if seeded:
            audit.record("egress.rate_seeded", {"prior_requests_in_window": seeded})
        self._client = httpx.Client(
            timeout=policy.REQUEST_TIMEOUT_S,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
            event_hooks={"request": [self._verify_wire_request]},
        )

    @property
    def request_count(self) -> int:
        """Number of requests issued so far; recorded in the snapshot manifest."""
        return self._request_count

    def get(
        self,
        capability_name: policy.CapabilityName,
        *,
        path_params: Mapping[str, str] | None = None,
        query_params: Mapping[str, str] | None = None,
    ) -> bytes:
        """Perform one allowlisted GET and return the body bytes.

        Raises :class:`EgressDenied` for any policy failure (audited) and
        :class:`EgressHTTPError` for a non-2xx vendor response.
        """
        capability = policy.CAPABILITIES[capability_name]
        try:
            url = build_url(capability, path_params or {}, query_params or {})
            authorised = authorise(policy.ALLOWED_METHOD, url)
            if authorised.capability.name is not capability_name:
                raise EgressDenied("built URL resolved to a different capability")
            if self._request_count >= policy.MAX_REQUESTS_PER_ACQUISITION:
                raise EgressDenied("per-acquisition request cap reached")
            SEAL.assert_data_network_allowed()
        except EgressDenied as exc:
            self._audit.record("egress.deny", {"capability": capability_name.value, "reason": str(exc)})
            raise

        self._throttle()
        waited = self._limiter.acquire()
        if waited >= 1.0:
            self._audit.record("egress.rate_wait", {"seconds": round(waited, 1), "requests_so_far": self._request_count})
        headers = {"Accept": "application/json"}
        if capability.attach_token:
            headers.update(self._token.bearer_header())
        self._request_count += 1
        self._audit.record("egress.allow", {"capability": capability_name.value, "url": url})

        with self._client.stream("GET", url, headers=headers) as response:
            if response.status_code != 200:
                self._audit.record(
                    "egress.http_error",
                    {"capability": capability_name.value, "status": response.status_code},
                )
                if response.status_code == 429:
                    raise EgressRateLimited(capability_name, response.status_code)
                raise EgressHTTPError(capability_name, response.status_code)
            chunks: list[bytes] = []
            size = 0
            # iter_bytes applies any Content-Encoding, so the cap bounds *decoded* size (bomb-safe).
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > policy.MAX_RESPONSE_BYTES:
                    self._audit.record("egress.deny", {"capability": capability_name.value, "reason": "response too large"})
                    raise EgressDenied(f"{capability_name.value}: response exceeds {policy.MAX_RESPONSE_BYTES} bytes")
                chunks.append(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        """Close the HTTP client and drop the token; called when the ETL seals its snapshot."""
        self._client.close()
        self._token.drop()
        self._audit.record("egress.closed", {"requests": self._request_count})

    def _throttle(self) -> None:
        """Sleep so consecutive requests are at least ``MIN_REQUEST_INTERVAL_S`` apart."""
        now = self._clock()
        if self._last_request_at is not None:
            wait = policy.MIN_REQUEST_INTERVAL_S - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._clock()

    def _verify_wire_request(self, request: httpx.Request) -> None:
        """httpx hook: re-authorise the request exactly as it will be sent (defence in depth)."""
        authorised = authorise(request.method, str(request.url))
        if "authorization" in request.headers and not authorised.capability.attach_token:
            raise EgressDenied(f"token must not be sent to {authorised.capability.host}")
