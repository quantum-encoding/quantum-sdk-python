"""Quantum AI API error types."""

from __future__ import annotations


class APIError(Exception):
    """Raised when the API responds with a non-2xx status code.

    The ``code`` attribute carries the stable machine-readable code from the
    backend's error taxonomy (see backend ``internal/server/errors.go``).
    Typed subclasses (InsufficientBalanceError, SpendCapExceededError, ...)
    are raised for the canonical billing codes so callers can catch by type
    instead of substring-matching messages.
    """

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

    def is_insufficient_balance(self) -> bool:
        """True if the account balance could not cover the request cost."""
        return self.status_code == 402 or self.code == "INSUFFICIENT_BALANCE"


# ---------------------------------------------------------------------------
# Typed billing / credit errors (codes from errors.go).
# ---------------------------------------------------------------------------


class InsufficientBalanceError(APIError):
    """402 INSUFFICIENT_BALANCE — wallet cannot cover the next step."""


class TrialExpiredError(APIError):
    """TRIAL_EXPIRED — trial credits exhausted."""


class SubscriptionLapsedError(APIError):
    """SUBSCRIPTION_LAPSED — subscription no longer active."""


class SpendCapExceededError(APIError):
    """SPEND_CAP_EXCEEDED — API-key spend cap hit."""


class BudgetFrozenError(APIError):
    """BUDGET_FROZEN — partner/operator budget kill-switch fired."""


class PaymentNotConfiguredError(APIError):
    """PAYMENT_NOT_CONFIGURED — operator billing misconfiguration."""


# Map of canonical backend code → typed exception class. Used by the client
# to raise the right subclass when parsing an error response. Keep in sync
# with internal/server/errors.go (CodeInsufficientBalance et al.).
_BILLING_ERROR_CODES: dict[str, type[APIError]] = {
    "INSUFFICIENT_BALANCE": InsufficientBalanceError,
    "TRIAL_EXPIRED": TrialExpiredError,
    "SUBSCRIPTION_LAPSED": SubscriptionLapsedError,
    "SPEND_CAP_EXCEEDED": SpendCapExceededError,
    "BUDGET_FROZEN": BudgetFrozenError,
    "PAYMENT_NOT_CONFIGURED": PaymentNotConfiguredError,
}


def _typed_error_for(status_code: int, code: str, message: str, request_id: str | None) -> APIError:
    """Return the typed APIError subclass for a code, falling back to APIError."""
    cls = _BILLING_ERROR_CODES.get(code)
    if cls is None:
        return APIError(status_code, code, message, request_id)
    return cls(status_code, code, message, request_id)


def is_rate_limit_error(err: BaseException) -> bool:
    """Check whether an error is a rate limit APIError."""
    return isinstance(err, APIError) and err.is_rate_limit()


def is_auth_error(err: BaseException) -> bool:
    """Check whether an error is an authentication APIError."""
    return isinstance(err, APIError) and err.is_auth()


def is_not_found_error(err: BaseException) -> bool:
    """Check whether an error is a not found APIError."""
    return isinstance(err, APIError) and err.is_not_found()


def is_insufficient_balance_error(err: BaseException) -> bool:
    """Check whether an error is an insufficient-balance APIError (402/INSUFFICIENT_BALANCE)."""
    return isinstance(err, APIError) and err.is_insufficient_balance()
