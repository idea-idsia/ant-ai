class ContextWindowExceededError(Exception):
    """The conversation no longer fits the model's context window.

    Backends raise this in place of their provider's own error (LiteLLM's
    `ContextWindowExceededError`, OpenAI's `BadRequestError` with code
    `context_length_exceeded`, ...) so the condition can be recognised without
    importing the provider. The provider's exception is chained as `__cause__`
    for logs.

    The message is written here, not copied from the provider, so it carries no
    prompt content or endpoint detail and is safe to forward to whoever called
    the agent — which `A2AExecutor` does.

    Args:
        model: The model that refused the request, kept for logs.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model
        super().__init__(
            "The conversation exceeds the model's context window. "
            "Start a new conversation or reduce the material attached to this one."
        )
