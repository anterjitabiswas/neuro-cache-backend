"""Adaptive Replacement Cache (ARC) implementation.

Reference:
  Nimrod Megiddo and Dharmendra S. Modha.
  "ARC: A Self-Tuning, Low Overhead Replacement Cache"
  USENIX Conference on File and Storage Technologies (FAST '03).
"""
from collections import OrderedDict
from typing import Any, Dict, Optional, Set

from policies.base import BasePolicy


class ARCPolicy(BasePolicy):
    """Adaptive Replacement Cache (ARC).

    Dynamically tunes the target cache allocation parameter `p` between
    recent pages (T1) and frequent pages (T2) using ghost histories (B1, B2).
    """

    def __init__(self, capacity: int = 10, name: str = "ARC") -> None:
        super().__init__(name=name)
        self.capacity = max(1, capacity)
        self.p: float = 0.0  # Target size of T1

        # T1: top part of L1 (in cache, accessed once recently)
        self.t1: OrderedDict[Any, None] = OrderedDict()
        # B1: bottom part of L1 (ghost cache for T1 evictions)
        self.b1: OrderedDict[Any, None] = OrderedDict()

        # T2: top part of L2 (in cache, accessed >= 2 times recently)
        self.t2: OrderedDict[Any, None] = OrderedDict()
        # B2: bottom part of L2 (ghost cache for T2 evictions)
        self.b2: OrderedDict[Any, None] = OrderedDict()

        self.last_victim_from: Optional[str] = None

    def _replace(self, item: Any) -> Any:
        """Helper to determine whether victim comes from T1 or T2."""
        len_t1 = len(self.t1)
        in_b2 = item in self.b2

        if len_t1 > 0 and ((in_b2 and len_t1 == int(self.p)) or (len_t1 > self.p)):
            # Evict from T1 (recency list)
            victim = next(iter(self.t1))
            self.last_victim_from = "T1"
            return victim
        else:
            # Evict from T2 (frequency list)
            if self.t2:
                victim = next(iter(self.t2))
                self.last_victim_from = "T2"
                return victim
            elif self.t1:
                victim = next(iter(self.t1))
                self.last_victim_from = "T1"
                return victim
            raise RuntimeError("ARC replace called with empty cache lists")

    def choose_victim(self, cache_items: Set[Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Select a victim according to ARC replace rules."""
        # Clean any stale items not in cache_items
        for item in list(self.t1.keys()):
            if item not in cache_items:
                del self.t1[item]
        for item in list(self.t2.keys()):
            if item not in cache_items:
                del self.t2[item]

        req_item = context.get("requested_item") if context else None
        victim = self._replace(req_item)
        return victim

    def on_hit(self, item: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """Handle cache hit: move item to MRU of T2."""
        if item in self.t1:
            del self.t1[item]
            self.t2[item] = None
        elif item in self.t2:
            self.t2.move_to_end(item)
        else:
            # In case of out-of-band hit
            self.t2[item] = None

    def on_miss(self, item: Any, evicted_item: Optional[Any], context: Optional[Dict[str, Any]] = None) -> None:
        """Handle cache miss: update p, maintain ghost queues, and insert new item."""
        target_list = "T1"

        # 1. Case B: Item is in B1 (ghost miss in recency history)
        if item in self.b1:
            delta1 = 1.0 if len(self.b1) >= len(self.b2) else (len(self.b2) / max(1, len(self.b1)))
            self.p = min(float(self.capacity), self.p + delta1)
            del self.b1[item]
            target_list = "T2"

        # 2. Case C: Item is in B2 (ghost miss in frequency history)
        elif item in self.b2:
            delta2 = 1.0 if len(self.b2) >= len(self.b1) else (len(self.b1) / max(1, len(self.b2)))
            self.p = max(0.0, self.p - delta2)
            del self.b2[item]
            target_list = "T2"

        # 3. Case D: Item is completely new (cold miss)
        else:
            len_l1 = len(self.t1) + len(self.b1)
            len_l2 = len(self.t2) + len(self.b2)

            if len_l1 == self.capacity:
                if len(self.t1) < self.capacity and self.b1:
                    self.b1.popitem(last=False)
            elif len_l1 < self.capacity and (len_l1 + len_l2) >= 2 * self.capacity:
                if self.b2:
                    self.b2.popitem(last=False)

            target_list = "T1"

        # 4. Handle eviction into ghost cache
        if evicted_item is not None:
            if self.last_victim_from == "T1":
                self.t1.pop(evicted_item, None)
                self.b1[evicted_item] = None
            else:
                self.t2.pop(evicted_item, None)
                self.b2[evicted_item] = None

        # 5. Insert new item into chosen list
        if target_list == "T2":
            self.t2[item] = None
        else:
            self.t1[item] = None

        # 6. Bounded ghost capacity checks: |B1| <= c, |B2| <= c
        while len(self.b1) > self.capacity:
            self.b1.popitem(last=False)
        while len(self.b2) > self.capacity:
            self.b2.popitem(last=False)

    def reset(self) -> None:
        """Reset ARC internal state."""
        self.p = 0.0
        self.t1.clear()
        self.b1.clear()
        self.t2.clear()
        self.b2.clear()
        self.last_victim_from = None

    def get_state(self) -> Dict[str, Any]:
        """Return diagnostic state."""
        return {
            "p": round(self.p, 3),
            "t1_size": len(self.t1),
            "b1_size": len(self.b1),
            "t2_size": len(self.t2),
            "b2_size": len(self.b2),
            "total_metadata_entries": len(self.t1) + len(self.b1) + len(self.t2) + len(self.b2),
        }
