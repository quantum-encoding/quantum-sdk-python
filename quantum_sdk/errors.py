"""Quantum AI API error types."""

from __future__ import annotations


class APIError(Exception):
    """Raised when the API responds with a non-2xx status code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        request_id: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        super().__init__(str(self))

    def __str__(self) -> str:
        base = f"qai: {self.status_code} {self.code}: {self.message}"
        if self.request_id:
            return f"{base} (request_id={self.request_id})"
        return base

    def is_rate_limit(self) -> bool:
        """True if the error is a 429 rate limit response."""
        return self.status_code == 429

    def is_auth(self) -> bool:
        """True if the error is a 401 or 403 authentication/authorization failure."""
        return self.status_code in (401, 403)

    def is_not_found(self) -> bool:
        """True if the error is a 404 not found response."""
        return self.status_code == 404


def is_rate_limit_error(err: BaseException) -> bool:
    """Check whether an error is a rate limit APIError."""
    return isinstance(err, APIError) and err.is_rate_limit()


def is_auth_error(err: BaseException) -> bool:
    """Check whether an error is an authentication APIError."""
    return isinstance(err, APIError) and err.is_auth()


def is_not_found_error(err: BaseException) -> bool:
    """Check whether an error is a not found APIError."""
    return isinstance(err, APIError) and err.is_not_found()
