"""Abstract base replacement policy interface for SMARTCACHE AI."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set


class BasePolicy(ABC):
    """Abstract interface for all cache replacement policies."""

    def __init__(self, name: str = "BasePolicy") -> None:
        self.name = name

    @abstractmethod
    def choose_victim(self, cache_items: Set[Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Select an item to evict from the currently cached items.

        Args:
            cache_items: The set of items currently residing in the cache.
            context: Optional dictionary containing request index, trace metadata, etc.

        Returns:
            The item chosen for eviction.
        """
        pass

    @abstractmethod
    def on_hit(self, item: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """Lifecycle hook invoked when a requested item is already in cache.

        Args:
            item: The requested item.
            context: Optional metadata dictionary.
        """
        pass

    @abstractmethod
    def on_miss(self, item: Any, evicted_item: Optional[Any], context: Optional[Dict[str, Any]] = None) -> None:
        """Lifecycle hook invoked when a requested item was absent from cache.

        Args:
            item: The requested item that caused a miss.
            evicted_item: The victim item that was evicted to make room (None if space was available).
            context: Optional metadata dictionary.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal policy state to initial conditions."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Return diagnostic or telemetry state for dashboard/logging.

        Returns:
            Dictionary containing policy-specific telemetry (e.g. weights, frequency maps).
        """
        return {}
