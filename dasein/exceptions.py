"""Dasein SDK exceptions."""


class DaseinError(Exception):
    """Base exception for all Dasein errors."""
    pass


class DaseinUnavailableError(DaseinError):
    """Raised when the service is temporarily unavailable (503)."""

    def __init__(self, message: str = "Service unavailable", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class DaseinRateLimitError(DaseinError):
    """Raised when rate limit is exceeded (429)."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class DaseinAuthError(DaseinError):
    """Raised on authentication failure (401/403)."""
    pass


class DaseinNotFoundError(DaseinError):
    """Raised when a resource is not found (404)."""
    pass


class DaseinBuildError(DaseinError):
    """Raised when a build fails."""
    pass
