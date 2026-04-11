"""Simple registry for student models."""

from __future__ import annotations

from typing import Callable, Dict

from .base import OverlapStudent


_REGISTRY: Dict[str, Callable[..., OverlapStudent]] = {}


def register_student(name: str) -> Callable[[Callable[..., OverlapStudent]], Callable[..., OverlapStudent]]:
    """Decorator to register a student builder under a string key."""

    def decorator(builder: Callable[..., OverlapStudent]) -> Callable[..., OverlapStudent]:
        key = str(name).lower()
        if key in _REGISTRY:
            raise ValueError(f"Student model '{key}' already registered")
        _REGISTRY[key] = builder
        return builder

    return decorator


def build_student(name: str, **kwargs) -> OverlapStudent:
    key = str(name).lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown student model '{key}'")
    return _REGISTRY[key](**kwargs)

