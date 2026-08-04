import asyncio
import unittest

from backend.app.retry import RetryPolicy, is_rate_limited, retry_after_seconds, retry_rate_limited


class RateLimitedError(Exception):
    status_code = 429


class RetryTests(unittest.TestCase):
    def test_retries_429_with_exponential_backoff(self) -> None:
        async def scenario() -> tuple[str, list[float], int]:
            attempts = 0
            delays: list[float] = []

            async def operation() -> str:
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise RateLimitedError("Too many requests")
                return "transcript"

            async def sleep(delay: float) -> None:
                delays.append(delay)

            result = await retry_rate_limited(
                operation,
                policy=RetryPolicy(max_attempts=4, base_delay_seconds=0.5, max_delay_seconds=5, jitter_ratio=0),
                sleep=sleep,
            )
            return result, delays, attempts

        result, delays, attempts = asyncio.run(scenario())
        self.assertEqual(result, "transcript")
        self.assertEqual(delays, [0.5, 1.0])
        self.assertEqual(attempts, 3)

    def test_honors_retry_after_header(self) -> None:
        error = RateLimitedError("rate limit")
        error.headers = {"retry-after": "3.5"}
        self.assertEqual(retry_after_seconds(error), 3.5)

    def test_converts_retry_after_milliseconds(self) -> None:
        error = RateLimitedError("429")
        error.headers = {"retry-after-ms": "250"}
        self.assertEqual(retry_after_seconds(error), 0.25)

    def test_non_rate_errors_are_not_retried(self) -> None:
        async def scenario() -> int:
            attempts = 0

            async def operation() -> str:
                nonlocal attempts
                attempts += 1
                raise ValueError("invalid request")

            with self.assertRaises(ValueError):
                await retry_rate_limited(operation, sleep=lambda _: asyncio.sleep(0))
            return attempts

        self.assertEqual(asyncio.run(scenario()), 1)

    def test_recognizes_throttling_messages(self) -> None:
        self.assertTrue(is_rate_limited(RateLimitedError("429")))
        self.assertTrue(is_rate_limited(RuntimeError("Request throttled by service")))
        self.assertFalse(is_rate_limited(ValueError("invalid audio format")))


if __name__ == "__main__":
    unittest.main()