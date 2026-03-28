"""Base schema utilities for the v0 scaffold."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class V0BaseModel(BaseModel):
    """Common base model for all structured runtime objects."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return self.model_dump(mode="json")

    def to_jsonable(self) -> dict[str, Any]:
        """Alias for explicit readability at call sites."""
        return self.to_dict()


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
