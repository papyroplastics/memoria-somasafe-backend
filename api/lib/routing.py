import functools
import inspect
from collections.abc import Callable
from typing import Any

from fastapi.routing import APIRoute
from sqlalchemy.orm import Session


def _closing_sessions(endpoint: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(endpoint)
    def call(*args: Any, **kwargs: Any) -> Any:
        try:
            return endpoint(*args, **kwargs)
        finally:
            for value in kwargs.values():
                if isinstance(value, Session):
                    value.close()
    call.closes_sessions = True  # type: ignore[attr-defined]
    return call


class SessionRoute(APIRoute):
    def __init__(self, path: str, endpoint: Callable[..., Any], **kwargs: Any) -> None:
        if not inspect.iscoroutinefunction(endpoint) and not getattr(endpoint, "closes_sessions", False):
            endpoint = _closing_sessions(endpoint)
        super().__init__(path, endpoint, **kwargs)
