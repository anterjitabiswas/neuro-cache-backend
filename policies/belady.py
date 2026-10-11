"""Belady's Optimal (MIN) offline oracle cache replacement policy."""
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Sequence, Set

from policies.base import BasePolicy


class BeladyPolicy(BasePolicy):
    """Offline optimal policy that evicts the item needed furthest in the future."""

    def __init__(self, trace: Optional[Sequence[Any]] = None, name: str = "Belady (Optimal)") -> None:
        super().__init__(name=name)
        self.trace: List[Any] = list(trace) if trace is not None else []
        self.occurrences: Dict[Any, deque[int]] = defaultdict(deque)
        self.current_step: int = 0
        if self.trace:
            self._preprocess_trace()

    def set_trace(self, trace: Sequence[Any]) -> None:
        """Provide or update the trace for offline analysis."""
        self.trace = list(trace)
        self._preprocess_trace()

    def _preprocess_trace(self) -> None:
        """Index all occurrences of each item in the trace for O(1) next-use lookups."""
        self.occurrences.clear()
        for idx, item in enumerate(self.trace):
            self.occurrences[item].append(idx)
        self.current_step = 0

    def choose_victim(self, cache_items: Set[Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Evict the item whose next occurrence in the trace is furthest in the future."""
        furthest_item = None
        max_next_use = -1

        for item in cache_items:
            # Clean past occurrences up to current_step
            queue = self.occurrences[item]
            while queue and queue[0] <= self.current_step:
                queue.popleft()

            if not queue:
                # Never used again, perfect candidate for eviction
                return item

            next_use = queue[0]
            if next_use > max_next_use:
                max_next_use = next_use
                furthest_item = item

        return furthest_item if furthest_item is not None else next(iter(cache_items))

    def on_hit(self, item: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """Advance current step counter."""
        self.current_step += 1
        queue = self.occurrences[item]
        while queue and queue[0] < self.current_step:
            queue.popleft()

    def on_miss(self, item: Any, evicted_item: Optional[Any], context: Optional[Dict[str, Any]] = None) -> None:
        """Advance current step counter."""
        self.current_step += 1
        queue = self.occurrences[item]
        while queue and queue[0] < self.current_step:
            queue.popleft()

    def reset(self) -> None:
        """Reset state and re-index trace."""
        self._preprocess_trace()

    def get_state(self) -> Dict[str, Any]:
        """Return diagnostic state."""
        return {
            "current_step": self.current_step,
            "total_trace_len": len(self.trace)
        }
