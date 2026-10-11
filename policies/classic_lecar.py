"""Classic / Vanilla LeCaR Policy baseline.

Reference:
  Viet Nguyen, Rishabh Iyer, et al.
  "LeCaR: A Learning-based Adaptive Cache Replacement Policy"
  10th USENIX Workshop on Hot Topics in Storage and File Systems (HotStorage '18).
"""
from collections import OrderedDict
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from policies.base import BasePolicy
from policies.lru import LRUPolicy
from policies.lfu import LFUPolicy


class ClassicLeCaRPolicy(BasePolicy):
    """Vanilla LeCaR policy as published in HotStorage '18.

    Uses fixed learning rate updates on all ghost hits without shift damping
    or temporal TTL expiration.
    """

    def __init__(
        self,
        capacity: int = 10,
        learning_rate: float = 0.25,
        discount: float = 1.0,
        history_multiplier: float = 1.0,
        seed: Optional[int] = 42,
        name: str = "LeCaR (Classic)",
    ) -> None:
        super().__init__(name=name)
        self.capacity = max(1, capacity)
        self.learning_rate = learning_rate
        self.discount = discount
        self.ghost_capacity = max(1, int(self.capacity * history_multiplier))
        self.rng = random.Random(seed)

        self.lru_expert = LRUPolicy(name="LRU_Expert")
        self.lfu_expert = LFUPolicy(name="LFU_Expert")

        self.h_lru: OrderedDict[Any, int] = OrderedDict()
        self.h_lfu: OrderedDict[Any, int] = OrderedDict()

        self.w_lru: float = 0.5
        self.w_lfu: float = 0.5
        self.current_step: int = 0
        self.last_chosen_expert: Optional[str] = None
        self.ghost_hits_lru: int = 0
        self.ghost_hits_lfu: int = 0
        self.weight_history: List[Tuple[int, float, float]] = [(0, 0.5, 0.5)]

    def choose_victim(self, cache_items: Set[Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Select a victim by polling experts and sampling by weights."""
        v_lru = self.lru_expert.choose_victim(cache_items, context)
        v_lfu = self.lfu_expert.choose_victim(cache_items, context)

        if v_lru == v_lfu:
            self.last_chosen_expert = "BOTH"
            return v_lru

        if self.rng.random() < self.w_lru:
            self.last_chosen_expert = "LRU"
            return v_lru
        else:
            self.last_chosen_expert = "LFU"
            return v_lfu

    def on_hit(self, item: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """Forward hit to sub-experts."""
        self.current_step += 1
        self.lru_expert.on_hit(item, context)
        self.lfu_expert.on_hit(item, context)
        self.weight_history.append((self.current_step, self.w_lru, self.w_lfu))

    def on_miss(self, item: Any, evicted_item: Optional[Any], context: Optional[Dict[str, Any]] = None) -> None:
        """Classic LeCaR update: immediately penalize on ghost hits."""
        self.current_step += 1
        updated = False

        if item in self.h_lru:
            evict_step = self.h_lru.pop(item)
            delta_t = max(0, self.current_step - evict_step)
            decay = self.discount ** delta_t if self.discount < 1.0 else 1.0
            self.w_lru *= math.exp(-self.learning_rate * decay)
            self.ghost_hits_lru += 1
            updated = True

        if item in self.h_lfu:
            evict_step = self.h_lfu.pop(item)
            delta_t = max(0, self.current_step - evict_step)
            decay = self.discount ** delta_t if self.discount < 1.0 else 1.0
            self.w_lfu *= math.exp(-self.learning_rate * decay)
            self.ghost_hits_lfu += 1
            updated = True

        if updated:
            self.w_lru = max(self.w_lru, 0.005)
            self.w_lfu = max(self.w_lfu, 0.005)
            total = self.w_lru + self.w_lfu
            self.w_lru /= total
            self.w_lfu /= total

        self.lru_expert.on_miss(item, evicted_item, context)
        self.lfu_expert.on_miss(item, evicted_item, context)

        if evicted_item is not None:
            if self.last_chosen_expert in ("LRU", "BOTH"):
                self.h_lru[evicted_item] = self.current_step
                while len(self.h_lru) > self.ghost_capacity:
                    self.h_lru.popitem(last=False)
            if self.last_chosen_expert in ("LFU", "BOTH"):
                self.h_lfu[evicted_item] = self.current_step
                while len(self.h_lfu) > self.ghost_capacity:
                    self.h_lfu.popitem(last=False)

        self.weight_history.append((self.current_step, self.w_lru, self.w_lfu))

    def reset(self) -> None:
        """Reset state."""
        self.lru_expert.reset()
        self.lfu_expert.reset()
        self.h_lru.clear()
        self.h_lfu.clear()
        self.w_lru = 0.5
        self.w_lfu = 0.5
        self.current_step = 0
        self.last_chosen_expert = None
        self.ghost_hits_lru = 0
        self.ghost_hits_lfu = 0
        self.weight_history = [(0, 0.5, 0.5)]

    def get_state(self) -> Dict[str, Any]:
        """Return diagnostic state."""
        return {
            "w_lru": round(self.w_lru, 4),
            "w_lfu": round(self.w_lfu, 4),
            "h_lru_size": len(self.h_lru),
            "h_lfu_size": len(self.h_lfu),
            "ghost_hits_lru": self.ghost_hits_lru,
            "ghost_hits_lfu": self.ghost_hits_lfu,
            "total_metadata_entries": self.capacity + len(self.h_lru) + len(self.h_lfu),
        }
