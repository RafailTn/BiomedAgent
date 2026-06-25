"""
Shared HTTP client for all biomedical REST-API tools.
=====================================================
One session, reused everywhere, that adds two things every tool was missing:

1. Caching  — repeated lookups (same gene, pathway, coordinates) hit a local
   SQLite cache instead of the network. Pathway names / gene coords basically
   never change, so this is a large speed + rate-limit win.
2. Retries  — transient 429/500/502/503/504 and timeouts are retried with
   exponential backoff instead of bubbling a one-shot failure up to the LLM
   (which then tends to hallucinate or give up).

Degrades gracefully: if `requests_cache` is unavailable the session still works,
just without persistence.
"""

import os
import logging
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load .env from this file's directory so the NCBI key is available regardless of
# import order or the process's working directory.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except Exception:
    pass

logger = logging.getLogger(__name__)

# Default timeout for every request unless a caller overrides it.
DEFAULT_TIMEOUT = 15

# Cache lives next to the code; 24h expiry is safe for the reference data we hit.
_CACHE_PATH = Path(__file__).parent / "http_cache"
_CACHE_EXPIRE_SECONDS = 24 * 60 * 60

# NCBI E-utilities allow 10 req/s with a key vs 3 without. Accept either name.
NCBI_API_KEY = os.getenv("PUBMED_API_KEY") or os.getenv("NCBI_API_KEY")


def _retry() -> Retry:
    return Retry(
        total=3,
        backoff_factor=0.5,                       # 0.5s, 1s, 2s
        status_forcelist=[429, 500, 502, 503, 504],
        # POST is included for read-only GraphQL queries (Open Targets); the repo
        # has no state-changing POSTs.
        allowed_methods=["GET", "POST"],
        respect_retry_after_header=True,
    )


def _build_session() -> requests.Session:
    try:
        from requests_cache import CachedSession

        session = CachedSession(
            cache_name=str(_CACHE_PATH),
            backend="sqlite",
            expire_after=_CACHE_EXPIRE_SECONDS,
            # Cache GraphQL POSTs too (matched on URL + body) — Open Targets
            # associations are stable day-to-day.
            allowable_methods=["GET", "POST"],
            stale_if_error=True,                  # serve stale on upstream failure
        )
        logger.info("HTTP cache enabled at %s.sqlite", _CACHE_PATH)
    except Exception as e:                        # missing lib, locked db, etc.
        logger.warning("HTTP cache unavailable (%s); using plain session.", e)
        session = requests.Session()

    adapter = HTTPAdapter(max_retries=_retry())
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# Single shared session for the whole process.
SESSION = _build_session()


def get(url: str, **kwargs) -> requests.Response:
    """Drop-in replacement for requests.get with shared cache + retries."""
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    return SESSION.get(url, **kwargs)


def post(url: str, **kwargs) -> requests.Response:
    """requests.post with shared cache + retries (for read-only GraphQL queries)."""
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    return SESSION.post(url, **kwargs)


def get_ncbi(url: str, params: dict | None = None, **kwargs) -> requests.Response:
    """requests.get for NCBI E-utilities, injecting the API key when available."""
    params = dict(params or {})
    if NCBI_API_KEY and "api_key" not in params:
        params["api_key"] = NCBI_API_KEY
    return get(url, params=params, **kwargs)
