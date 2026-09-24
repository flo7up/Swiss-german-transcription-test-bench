"""Retry helpers for transient model-service throttling."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
import random
from typing import TypeVar


LOGGER = logging.getLogger(__name__)
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 6
    base_delay_seconds: float = 0.75
    max_delay_seconds: float = 16.0
    jitter_ratio: float = 0.2


def is_rate_limited(error: Exception) -> bool:
    """Identify 429 or throttling errors across HTTP and WebSocket client types."""
    response = getattr(error, "response", None)
    statuses = (getattr(error, "status_code", None), getattr(error, "status", None), getattr(response, "status_code", None))
    if 429 in statuses:
        return True
    message = str(error).casefold()
    return "429" in message or "rate limit" in message or "too many requests" in message or "throttl" in message


def retry_after_seconds(error: Exception) -> float | None:
    """Return the service-provided Retry-After delay when available."""
    response = getattr(error, "response", None)
    headers = getattr(error, "headers", None) or getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    is_millisecond_value = value is None
    if value is None:
        value = headers.get("retry-after-ms") or headers.get("Retry-After-Ms")
    try:
        delay = float(value)
    except (TypeError, ValueError):
        return None
    if is_millisecond_value:
        delay /= 1_000
    return delay if delay > 0 else None


async def retry_rate_limited(
    operation: Callable[[], Awaitable[ResultT]],
    *,
    policy: RetryPolicy = RetryPolicy(),
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    random_value: Callable[[], float] = random.random,
) -> ResultT:
    """Run an idempotent model operation with bounded exponential 429 retries."""
    if policy.max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(policy.max_attempts):
        try:
            return await operation()
        except Exception as error:
            if not is_rate_limited(error) or attempt == policy.max_attempts - 1:
                raise

            retry_after = retry_after_seconds(error)
            if retry_after is None:
                base_delay = min(policy.max_delay_seconds, policy.base_delay_seconds * (2**attempt))
                delay = min(policy.max_delay_seconds, base_delay * (1 + policy.jitter_ratio * random_value()))
            else:
                delay = retry_after
            LOGGER.warning(
                "Rate limited by model service; retrying attempt %s of %s in %.2f seconds.",
                attempt + 2,
                policy.max_attempts,
                delay,
            )
            await sleep(delay)

    raise RuntimeError("Rate-limit retry loop exited unexpectedly")