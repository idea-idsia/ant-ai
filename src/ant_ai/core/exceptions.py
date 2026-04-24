class HookBlockedError(Exception):
    """Raised when a hook returns BLOCK — the response must not be used."""


class HookMaxRetriesError(Exception):
    """Raised when the retry loop exhausts all attempts and the hook still fails."""
