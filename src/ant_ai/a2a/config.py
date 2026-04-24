from httpx import Timeout
from pydantic import BaseModel, ConfigDict, Field, field_validator

type TimeoutTypes = (
    float
    | None
    | tuple[float | None, float | None, float | None, float | None]
    | Timeout
)


class A2AConfig(BaseModel):
    """Configuration used to set the connection with the A2A server."""

    endpoint: str = Field(description="The URL of the A2A server to connect to.")
    timeout: TimeoutTypes = Field(
        default_factory=lambda: Timeout(connect=10, read=None, write=10, pool=10),
        description=(
            "Timeout configuration for the A2A client. Can be a float (total timeout) or a dict with "
            "connect/read/write/pool timeouts. See httpx.Timeout for more details."
        ),
    )

    agent_card_path: str = Field(
        default="/.well-known/agent-card.json",
        description="The path, on the remote url, to the agent card.",
    )
    supported_protocol_bindings: tuple[str, ...] = Field(
        default=("JSONRPC", "HTTP+JSON"),
        description="The supported A2A protocol bindings.",
    )
    streaming: bool = Field(default=True, description="Whether to enable streaming.")
    propagate_trace_context: bool = Field(
        default=True,
        description=(
            "Whether to inject the current trace context into outbound A2A requests. Set to False when calling third-party agents you do not own."
        ),
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("timeout", mode="before")
    @classmethod
    def normalize_timeout(cls, value):
        match value:
            case Timeout():
                return value
            case int() | float():
                return Timeout(float(value))
            case tuple():
                return Timeout(value)
            case dict():
                return Timeout(**value)
            case _:
                raise TypeError(f"Invalid timeout value: {value!r}")
