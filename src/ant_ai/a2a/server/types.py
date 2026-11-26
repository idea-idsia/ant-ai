from pydantic import BaseModel, ConfigDict

from ant_ai.core.events import AnyEvent


class A2AMetadata(BaseModel):
    """Metadata for A2A interactions"""

    model_config = ConfigDict(frozen=True)

    event: AnyEvent | None = None
