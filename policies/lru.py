"""Least Recently Used (LRU) baseline cache replacement policy."""
from collections import OrderedDict
from typing import Any, Dict, Optional, Set

from policies.base import BasePolicy


class LRUPolicy(BasePolicy):
    """LRU eviction policy using OrderedDict for O(1) recency tracking."""

    def __init__(self, name: str = "LRU") -> None:
        super().__init__(name=name)
        # OrderedDict stores items ordered from LRU (head) to MRU (tail)
        self.order: OrderedDict[Any, None] = OrderedDict()

    def choose_victim(self, cache_items: Set[Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Select the least recently used item among cached items."""
        # Clean up any items no longer in cache_items
        for item in list(self.order.keys()):
            if item not in cache_items:
                del self.order[item]

        for item in self.order:
            if item in cache_items:
                return item

        # Fallback if empty or out of sync: return any element from cache_items
        return next(iter(cache_items))

    def on_hit(self, item: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """Move accessed item to MRU (tail)."""
        if item in self.order:
            self.order.move_to_end(item)
        else:
            self.order[item] = None

    def on_miss(self, item: Any, evicted_item: Optional[Any], context: Optional[Dict[str, Any]] = None) -> None:
        """Handle miss: remove evicted item and append new item to MRU."""
        if evicted_item is not None and evicted_item in self.order:
            del self.order[evicted_item]
        self.order[item] = None

    def reset(self) -> None:
        """Clear LRU order."""
        self.order.clear()

    def get_state(self) -> Dict[str, Any]:
        """Return diagnostic state."""
        return {
            "lru_order": list(self.order.keys())
        }
