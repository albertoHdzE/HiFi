"""Transient-network retry for broker/data calls (DJ-117).

A brief connectivity blip should not waste a nightly run. The per-ticker OHLCV
fetch already tolerated errors, but the Alpaca account calls (connect, positions,
orders) were unwrapped — a dropped connection there crashed the whole cycle
(rc=1). This adds bounded exponential-backoff retry around network calls, and
only for genuinely transient network errors (not logic bugs).
"""

from __future__ import annotations

import functools
import logging
import time

logger = logging.getLogger(__name__)

# Transient network failures worth retrying. Deliberately network-only so real
# bugs (KeyError, ValueError, auth 4xx) surface immediately instead of looping.
try:
    import requests  # noqa: PLC0415
    _REQUESTS_ERR: tuple = (requests.exceptions.RequestException,)
except Exception:  # pragma: no cover
    _REQUESTS_ERR = ()

NET_ERRORS: tuple = _REQUESTS_ERR + (ConnectionError, TimeoutError, OSError)


def with_retry(attempts: int = 4, base_delay: float = 3.0,
               exceptions: tuple = NET_ERRORS):
    """Retry a function on transient network errors with exponential backoff.

    attempts=4, base_delay=3s -> waits 3s, 6s, 12s between the 4 tries
    (~21s total) before giving up and re-raising. Bounded so a real outage
    still fails in reasonable time rather than hanging forever.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for i in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if i == attempts - 1:
                        raise
                    delay = base_delay * (2 ** i)
                    logger.warning(
                        "%s network error (%s); retry %d/%d in %.0fs",
                        getattr(fn, "__name__", "call"), exc, i + 1, attempts - 1, delay,
                    )
                    time.sleep(delay)
        return wrapper
    return deco
