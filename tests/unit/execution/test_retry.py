"""Tests for the transient-network retry helper (DJ-117)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from hifi.execution.retry import with_retry


def test_succeeds_first_try():
    calls = []

    @with_retry(attempts=3, base_delay=0)
    def f():
        calls.append(1)
        return "ok"

    assert f() == "ok"
    assert len(calls) == 1


def test_retries_then_succeeds():
    calls = []

    @with_retry(attempts=4, base_delay=0)
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("blip")
        return "ok"

    with patch("time.sleep"):
        assert f() == "ok"
    assert len(calls) == 3


def test_gives_up_after_attempts():
    @with_retry(attempts=3, base_delay=0)
    def f():
        raise TimeoutError("down")

    with patch("time.sleep"), pytest.raises(TimeoutError):
        f()


def test_non_network_error_not_retried():
    calls = []

    @with_retry(attempts=3, base_delay=0)
    def f():
        calls.append(1)
        raise ValueError("bug")

    with pytest.raises(ValueError):
        f()
    assert len(calls) == 1  # not retried
