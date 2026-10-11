"""LeCaR-style Adaptive Cache Replacement Policy for SMARTCACHE AI.

Combines LRU and LFU experts and uses dual shadow (ghost) caches
to perform regret-based Exponential Gradient (Exp3/Hedge) updates.
Enhanced with workload-shift detection, transition metrics tracking,
and audited ghost-cache credit attribution.
"""
from collections import deque, OrderedDict
import math
import random
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from policies.base import BasePolicy
from policies.lru import LRUPolicy
from policies.lfu import LFUPolicy


class LeCaRAdaptivePolicy(BasePolicy):
    """Adaptive cache replacement policy using LeCaR online learning.

    Maintains dual ghost caches (H_LRU, H_LFU) to attribute eviction mistakes.
    Features:
      1. Audited ghost-cache attribution (TTL expiration, mutually exclusive histories,
         consensus neutrality, duplicate/cycle safety).
      2. Workload-shift detection (sliding window hit rate, reuse density, regret momentum)
         to avoid knee-jerk switching on isolated misses.
      3. Measurable policy switching (adaptation latency and extra transition misses).
    """

    def __init__(
        self,
        capacity: int = 10,
        learning_rate: Optional[float] = None,
        discount: float = 1.0,
        history_multiplier: float = 1.0,
        ttl_multiplier: float = 3.0,
        damping_factor: float = 0.35,
        shift_threshold: float = 0.20,
        epsilon_min: float = 0.005,
        deterministic: bool = False,
        seed: Optional[int] = 42,
        name: str = "SmartCache (LeCaR)",
    ) -> None:
        super().__init__(name=name)
        self.capacity = max(1, capacity)
        # Adaptive base learning rate sqrt(ln(2) / capacity) if not explicitly set
        self.learning_rate = (
            learning_rate
            if learning_rate is not None
            else min(0.45, max(0.05, math.sqrt(math.log(2.0) / self.capacity)))
        )
        self.discount = discount
        self.ghost_capacity = max(1, int(self.capacity * history_multiplier))
        self.ghost_ttl = max(10, int(self.capacity * ttl_multiplier))
        self.damping_factor = damping_factor
        self.shift_threshold = shift_threshold
        self.epsilon_min = epsilon_min
        self.deterministic = deterministic
        self.rng = random.Random(seed)

        # Experts
        self.lru_expert = LRUPolicy(name="LRU_Expert")
        self.lfu_expert = LFUPolicy(name="LFU_Expert")

        # Ghost history caches: item -> eviction_step
        self.h_lru: OrderedDict[Any, int] = OrderedDict()
        self.h_lfu: OrderedDict[Any, int] = OrderedDict()

        # Expert weights (w_lru, w_lfu), normalized to sum to 1.0
        self.w_lru: float = 0.5
        self.w_lfu: float = 0.5

        # Tracking variables
        self.current_step: int = 0
        self.last_chosen_expert: Optional[str] = None
        self.ghost_hits_lru: int = 0
        self.ghost_hits_lfu: int = 0
        self.expired_ghost_entries: int = 0

        # Workload-shift detection state
        self.window_size: int = max(20, 2 * self.capacity)
        self.recent_outcomes: Deque[Tuple[bool, Any]] = deque(maxlen=self.window_size)
        self.recent_ghost_hits: Deque[Tuple[int, str]] = deque(maxlen=self.window_size)
        self.baseline_hit_rate: float = 0.5
        self.is_shift_active: bool = False

        # Transition measurement state
        self.active_transition: Optional[Dict[str, Any]] = None
        self.completed_transitions: List[Dict[str, Any]] = []

        # Telemetry history for dashboard: (step, w_lru, w_lfu)
        self.weight_history: List[Tuple[int, float, float]] = [(0, 0.5, 0.5)]

    def choose_victim(self, cache_items: Set[Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Select a victim by consulting LRU and LFU experts and sampling by weights."""
        v_lru = self.lru_expert.choose_victim(cache_items, context)
        v_lfu = self.lfu_expert.choose_victim(cache_items, context)

        if v_lru == v_lfu:
            # Both experts agree: consensus eviction
            self.last_chosen_expert = "CONSENSUS"
            return v_lru

        # Experts disagree: sample based on weights or choose argmax
        if self.deterministic:
            if self.w_lru >= self.w_lfu:
                chosen = "LRU"
                victim = v_lru
            else:
                chosen = "LFU"
                victim = v_lfu
        else:
            if self.rng.random() < self.w_lru:
                chosen = "LRU"
                victim = v_lru
            else:
                chosen = "LFU"
                victim = v_lfu

        self.last_chosen_expert = chosen
        return victim

    def on_hit(self, item: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """Forward hit to both experts and purge item from ghost histories."""
        self.current_step += 1
        
        # Invariant: Item in cache must NEVER reside in ghost caches
        self.h_lru.pop(item, None)
        self.h_lfu.pop(item, None)

        # Update workload tracker
        self.recent_outcomes.append((True, item))
        self.baseline_hit_rate = 0.95 * self.baseline_hit_rate + 0.05 * 1.0

        # Update transition tracking
        self._check_transition_status(is_miss=False)

        # Forward hit to sub-experts
        self.lru_expert.on_hit(item, context)
        self.lfu_expert.on_hit(item, context)
        self.weight_history.append((self.current_step, self.w_lru, self.w_lfu))

    def on_miss(self, item: Any, evicted_item: Optional[Any], context: Optional[Dict[str, Any]] = None) -> None:
        """Handle miss: attribute regret with TTL & shift awareness, update weights, then insert."""
        self.current_step += 1
        self.recent_outcomes.append((False, item))
        self.baseline_hit_rate = 0.95 * self.baseline_hit_rate + 0.05 * 0.0

        # 1. Detect whether a true workload shift is underway
        self._evaluate_workload_shift()

        # 2. Check for transition initiation or ongoing transition accounting
        self._check_transition_status(is_miss=True)

        # 3. Regret attribution with TTL check and shift-moderated learning rate
        updated = False
        dominant_expert = "LRU" if self.w_lru >= self.w_lfu else "LFU"

        # Check LRU ghost cache
        if item in self.h_lru:
            evict_step = self.h_lru.pop(item)
            delta_t = max(0, self.current_step - evict_step)

            if delta_t > self.ghost_ttl:
                # Expired ghost entry: access is too distant in time to be an eviction error
                self.expired_ghost_entries += 1
            else:
                eff_lr = self._compute_effective_learning_rate(target_expert="LRU", dominant_expert=dominant_expert)
                decay = self.discount ** delta_t if self.discount < 1.0 else 1.0
                self.w_lru *= math.exp(-eff_lr * decay)
                self.ghost_hits_lru += 1
                self.recent_ghost_hits.append((self.current_step, "LRU"))
                updated = True

        # Check LFU ghost cache
        if item in self.h_lfu:
            evict_step = self.h_lfu.pop(item)
            delta_t = max(0, self.current_step - evict_step)

            if delta_t > self.ghost_ttl:
                # Expired ghost entry: access is too distant in time to be an eviction error
                self.expired_ghost_entries += 1
            else:
                eff_lr = self._compute_effective_learning_rate(target_expert="LFU", dominant_expert=dominant_expert)
                decay = self.discount ** delta_t if self.discount < 1.0 else 1.0
                self.w_lfu *= math.exp(-eff_lr * decay)
                self.ghost_hits_lfu += 1
                self.recent_ghost_hits.append((self.current_step, "LFU"))
                updated = True

        # Cleanse item from both ghost caches before inserting into active cache
        self.h_lru.pop(item, None)
        self.h_lfu.pop(item, None)

        # 4. Normalize weights with numerical stability safeguards
        if updated:
            self._normalize_weights()

        # 5. Inform both experts
        self.lru_expert.on_miss(item, evicted_item, context)
        self.lfu_expert.on_miss(item, evicted_item, context)

        # 6. Record evicted item into appropriate ghost cache
        if evicted_item is not None:
            if self.last_chosen_expert == "LRU":
                # Ensure evicted item is mutually exclusive: remove from LFU ghost if present
                self.h_lfu.pop(evicted_item, None)
                self._add_to_ghost(self.h_lru, evicted_item)
            elif self.last_chosen_expert == "LFU":
                # Ensure evicted item is mutually exclusive: remove from LRU ghost if present
                self.h_lru.pop(evicted_item, None)
                self._add_to_ghost(self.h_lfu, evicted_item)
            elif self.last_chosen_expert == "CONSENSUS":
                # Consensus eviction: neither expert was uniquely wrong.
                # Remove from both ghosts so neither expert is penalised.
                self.h_lru.pop(evicted_item, None)
                self.h_lfu.pop(evicted_item, None)

        self.weight_history.append((self.current_step, self.w_lru, self.w_lfu))

    def _compute_effective_learning_rate(self, target_expert: str, dominant_expert: str) -> float:
        """Compute effective learning rate, dampening single isolated misses during steady state."""
        if self.is_shift_active:
            # Full learning rate during active workload transitions for fast adaptation
            return self.learning_rate

        # Count recent ghost hits on the target expert
        recent_regrets = sum(1 for _, exp in self.recent_ghost_hits if exp == target_expert)

        if target_expert == dominant_expert:
            if recent_regrets <= 1:
                # Isolated unlucky miss on dominant expert: apply damping factor
                return self.learning_rate * self.damping_factor
            else:
                # Sustained regret on dominant expert: apply full rate
                return self.learning_rate
        else:
            # Penalizing non-dominant expert can be full rate
            return self.learning_rate

    def _evaluate_workload_shift(self) -> None:
        """Evaluate recent hit rate and reuse density to detect workload shifts."""
        if len(self.recent_outcomes) < max(10, self.capacity):
            self.is_shift_active = False
            return

        recent_hits = sum(1 for hit, _ in self.recent_outcomes if hit)
        recent_hr = recent_hits / len(self.recent_outcomes)

        unique_items = len(set(item for _, item in self.recent_outcomes))
        unique_density = unique_items / len(self.recent_outcomes)

        # 1. Substantial drop in hit rate below baseline
        hit_rate_drop = (self.baseline_hit_rate - recent_hr) >= self.shift_threshold

        # 2. Sequential scan or burst pattern (high unique density with zero/low hits)
        scan_detected = (unique_density >= 0.85) and (recent_hr < 0.15)

        # 3. Sustained regret on current dominant expert
        dominant_expert = "LRU" if self.w_lru >= self.w_lfu else "LFU"
        recent_dom_regrets = sum(1 for _, exp in self.recent_ghost_hits if exp == dominant_expert)
        sustained_regret = recent_dom_regrets >= 2

        self.is_shift_active = hit_rate_drop or scan_detected or sustained_regret

    def _check_transition_status(self, is_miss: bool) -> None:
        """Track policy adaptation latency and count extra misses during transitions."""
        current_dominant = "LRU" if self.w_lru >= 0.5 else "LFU"
        recent_hr = (
            sum(1 for h, _ in self.recent_outcomes if h) / len(self.recent_outcomes)
            if self.recent_outcomes
            else 0.5
        )

        if self.active_transition is None:
            # Check if a transition is initiating
            if self.is_shift_active or abs(self.w_lru - 0.5) < 0.05:
                # Transition started
                from_expert = "LFU" if current_dominant == "LRU" else "LRU"
                self.active_transition = {
                    "transition_id": len(self.completed_transitions) + 1,
                    "start_step": self.current_step,
                    "end_step": None,
                    "duration_requests": 0,
                    "from_expert": from_expert,
                    "to_expert": current_dominant,
                    "trigger_reason": "WORKLOAD_SHIFT" if self.is_shift_active else "WEIGHT_CROSSOVER",
                    "pre_transition_hit_rate": round(recent_hr, 4),
                    "misses_during_transition": 1 if is_miss else 0,
                    "extra_misses": 0,
                    "stabilized": False,
                }
        else:
            # Currently inside a transition
            if is_miss:
                self.active_transition["misses_during_transition"] += 1

            duration = self.current_step - self.active_transition["start_step"]
            dom_weight = max(self.w_lru, self.w_lfu)

            # Check for stabilization:
            # New expert has reached clear dominance (weight >= 0.65) and duration >= 5
            if (dom_weight >= 0.65 and duration >= 5) or duration >= (self.capacity * 10):
                expected_misses = int(duration * (1.0 - self.active_transition["pre_transition_hit_rate"]))
                actual_misses = self.active_transition["misses_during_transition"]
                extra_misses = max(0, actual_misses - expected_misses)

                self.active_transition["end_step"] = self.current_step
                self.active_transition["duration_requests"] = duration
                self.active_transition["to_expert"] = current_dominant
                self.active_transition["extra_misses"] = extra_misses
                self.active_transition["stabilized"] = True

                self.completed_transitions.append(dict(self.active_transition))
                self.active_transition = None

    def _add_to_ghost(self, ghost_dict: OrderedDict[Any, int], item: Any) -> None:
        """Add an item to a ghost cache, maintaining bounded capacity and purging expired items."""
        if item in ghost_dict:
            ghost_dict.move_to_end(item)
        ghost_dict[item] = self.current_step

        # Enforce capacity bound
        while len(ghost_dict) > self.ghost_capacity:
            ghost_dict.popitem(last=False)

        # Lazy purge of oldest expired items
        while ghost_dict:
            oldest_item, oldest_step = next(iter(ghost_dict.items()))
            if self.current_step - oldest_step > self.ghost_ttl:
                ghost_dict.pop(oldest_item)
                self.expired_ghost_entries += 1
            else:
                break

    def _normalize_weights(self) -> None:
        """Enforce weight bounds [epsilon_min, 1 - epsilon_min] and sum-to-1 normalization."""
        self.w_lru = max(self.w_lru, self.epsilon_min)
        self.w_lfu = max(self.w_lfu, self.epsilon_min)

        total = self.w_lru + self.w_lfu
        if math.isnan(total) or total <= 0:
            self.w_lru = 0.5
            self.w_lfu = 0.5
        else:
            self.w_lru = self.w_lru / total
            self.w_lfu = self.w_lfu / total

    def reset(self) -> None:
        """Reset policy to initial state."""
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
        self.expired_ghost_entries = 0
        self.recent_outcomes.clear()
        self.recent_ghost_hits.clear()
        self.baseline_hit_rate = 0.5
        self.is_shift_active = False
        self.active_transition = None
        self.completed_transitions.clear()
        self.weight_history = [(0, 0.5, 0.5)]

    def get_state(self) -> Dict[str, Any]:
        """Return diagnostic telemetry including transition and shift metrics."""
        avg_latency = (
            sum(t["duration_requests"] for t in self.completed_transitions) / len(self.completed_transitions)
            if self.completed_transitions
            else 0.0
        )
        total_extra_misses = sum(t["extra_misses"] for t in self.completed_transitions)

        return {
            "w_lru": round(self.w_lru, 4),
            "w_lfu": round(self.w_lfu, 4),
            "h_lru_size": len(self.h_lru),
            "h_lfu_size": len(self.h_lfu),
            "ghost_hits_lru": self.ghost_hits_lru,
            "ghost_hits_lfu": self.ghost_hits_lfu,
            "expired_ghost_entries": self.expired_ghost_entries,
            "last_chosen_expert": self.last_chosen_expert,
            "is_shift_active": self.is_shift_active,
            "completed_transitions": len(self.completed_transitions),
            "avg_adaptation_latency": round(avg_latency, 2),
            "total_extra_misses": total_extra_misses,
        }
