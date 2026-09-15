"""Compatibility entry point: original baseline API is preserved."""
from .baseline import AddressMatcher, MatchResult, evaluate_at_k

__all__ = ["AddressMatcher", "MatchResult", "evaluate_at_k"]
