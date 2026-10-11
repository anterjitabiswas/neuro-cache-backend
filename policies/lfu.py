"""Least Frequently Used (LFU) baseline cache replacement policy."""
from collections import defaultdict, OrderedDict
from typing import Any, Dict, Optional, Set

from policies.base import BasePolicy


class LFUPolicy(BasePolicy):
    """O(1) LFU eviction policy with deterministic LRU tie-breaking for equal frequencies."""

    def __init__(self, name: str = "LFU") -> None:
        super().__init__(name=name)
        # item -> frequency count
        self.freq_map: Dict[Any, int] = {}
        # frequency -> OrderedDict of items (LRU at head, MRU at tail)
        self.freq_buckets: Dict[int, OrderedDict[Any, None]] = defaultdict(OrderedDict)
        self.min_freq: int = 0

    def choose_victim(self, cache_items: Set[Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Select the item with lowest frequency, breaking ties using LRU."""
        # Clean up any items that are no longer in cache_items
        for item in list(self.freq_map.keys()):
            if item not in cache_items:
                f = self.freq_map.pop(item)
                if f in self.freq_buckets and item in self.freq_buckets[f]:
                    del self.freq_buckets[f][item]

        # Ensure min_freq points to a non-empty bucket
        if self.min_freq not in self.freq_buckets or not self.freq_buckets[self.min_freq]:
            active_freqs = [f for f, bucket in self.freq_buckets.items() if bucket]
            if active_freqs:
                self.min_freq = min(active_freqs)
            else:
                return next(iter(cache_items))

        # Evict head of min_freq bucket (least recently used among minimum frequency)
        victim = next(iter(self.freq_buckets[self.min_freq]))
        return victim

    def on_hit(self, item: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """Increment frequency and advance item to higher frequency bucket."""
        freq = self.freq_map.get(item, 0)
        new_freq = freq + 1
        self.freq_map[item] = new_freq

        # Remove from old bucket
        if freq in self.freq_buckets and item in self.freq_buckets[freq]:
            del self.freq_buckets[freq][item]
            if not self.freq_buckets[freq] and self.min_freq == freq:
                self.min_freq = new_freq

        # Add to new bucket as MRU
        self.freq_buckets[new_freq][item] = None

    def on_miss(self, item: Any, evicted_item: Optional[Any], context: Optional[Dict[str, Any]] = None) -> None:
        """Handle miss: clean up evicted item and register new item with initial frequency 1."""
        if evicted_item is not None:
            f = self.freq_map.pop(evicted_item, None)
            if f is not None and f in self.freq_buckets and evicted_item in self.freq_buckets[f]:
                del self.freq_buckets[f][evicted_item]

        # Insert new item with freq = 1
        self.freq_map[item] = 1
        self.freq_buckets[1][item] = None
        self.min_freq = 1

    def reset(self) -> None:
        """Reset frequency tracking state."""
        self.freq_map.clear()
        self.freq_buckets.clear()
        self.min_freq = 0

    def get_state(self) -> Dict[str, Any]:
        """Return diagnostic state."""
        return {
            "min_freq": self.min_freq,
            "freq_map": dict(self.freq_map)
        }
