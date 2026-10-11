"""SIEVE Cache Replacement Algorithm implementation.

Reference:
  Yazhuo Zhang, Yazhuo Zhang, et al.
  "SIEVE is Simpler than LRU: an Efficient Cache Replacement Policy"
  21st USENIX Symposium on Networked Systems Design and Implementation (NSDI '24).
"""
from typing import Any, Dict, Optional, Set

from policies.base import BasePolicy


class _SieveNode:
    """Internal doubly linked list node for SIEVE."""
    __slots__ = ("key", "visited", "prev", "next")

    def __init__(self, key: Any) -> None:
        self.key: Any = key
        self.visited: bool = False
        self.prev: Optional["_SieveNode"] = None
        self.next: Optional["_SieveNode"] = None


class SIEVEPolicy(BasePolicy):
    """SIEVE replacement policy (NSDI '24).

    Uses a single FIFO queue with a visited bit and a moving hand pointer.
    On eviction, scans backwards from `hand`, resetting visited bits until
    finding an unvisited victim.
    """

    def __init__(self, capacity: int = 10, name: str = "SIEVE") -> None:
        super().__init__(name=name)
        self.capacity = max(1, capacity)
        self.nodes: Dict[Any, _SieveNode] = {}
        self.head: Optional[_SieveNode] = None  # Most recently inserted
        self.tail: Optional[_SieveNode] = None  # Oldest inserted
        self.hand: Optional[_SieveNode] = None  # Eviction hand pointer

    def _remove_node(self, node: _SieveNode) -> None:
        """Unlink a node from the doubly linked list."""
        if node == self.hand:
            self.hand = node.prev

        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

        node.prev = None
        node.next = None

    def _insert_head(self, node: _SieveNode) -> None:
        """Insert node at the head of the queue."""
        node.next = self.head
        node.prev = None
        if self.head:
            self.head.prev = node
        self.head = node
        if self.tail is None:
            self.tail = node

    def choose_victim(self, cache_items: Set[Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Select a victim by advancing the hand pointer and checking visited flags."""
        curr = self.hand if self.hand is not None else self.tail

        if curr is None:
            return next(iter(cache_items))

        # Scan backwards, clearing visited bits
        attempts = 0
        max_attempts = max(1, len(self.nodes) * 2)

        while curr and curr.visited and attempts < max_attempts:
            curr.visited = False
            curr = curr.prev if curr.prev is not None else self.tail
            attempts += 1

        if curr is None:
            curr = self.tail

        victim_key = curr.key if curr else next(iter(cache_items))
        self.hand = curr.prev if curr else None
        return victim_key

    def on_hit(self, item: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """Mark visited bit to True upon hit."""
        node = self.nodes.get(item)
        if node is not None:
            node.visited = True

    def on_miss(self, item: Any, evicted_item: Optional[Any], context: Optional[Dict[str, Any]] = None) -> None:
        """Handle miss: clean up evicted node and insert new item at head with visited=False."""
        if evicted_item is not None and evicted_item in self.nodes:
            ev_node = self.nodes.pop(evicted_item)
            self._remove_node(ev_node)

        new_node = _SieveNode(key=item)
        new_node.visited = False
        self.nodes[item] = new_node
        self._insert_head(new_node)

    def reset(self) -> None:
        """Reset internal queue and hand pointer."""
        self.nodes.clear()
        self.head = None
        self.tail = None
        self.hand = None

    def get_state(self) -> Dict[str, Any]:
        """Return diagnostic state."""
        return {
            "queue_size": len(self.nodes),
            "hand_key": self.hand.key if self.hand else None,
            "total_metadata_entries": len(self.nodes),
        }
