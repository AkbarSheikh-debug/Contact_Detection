"""
Impact event classifier and session logger.
Aggregates per-frame contact features into labelled impact events with timing.
"""
from dataclasses import dataclass

from .pair_analyzer import ContactFeatures
from config import IMPACT_COOLDOWN_FRAMES, GLOBAL_COOLDOWN_FRAMES


@dataclass
class ImpactEvent:
    frame: int
    time_sec: float
    aggressor_id: int
    receiver_id: int
    contact_point: list
    contact_region: str      # "head" | "torso"
    striking_limb: str       # "left_jab" | "right_cross" | …
    probability: float
    velocity: float
    impact_type: str         # "head_impact" | "torso_impact"

    @property
    def label(self) -> str:
        clean = self.striking_limb.replace("_", " ").title()
        return f"{clean} -> {self.contact_region.title()}"


class ImpactClassifier:
    """
    Wraps PairInteractionAnalyzer results to deduplicate and log impact events.

    Two-level cooldown:
      Per-pair  (IMPACT_COOLDOWN_FRAMES)  — same pair can't fire again too quickly.
      Global    (GLOBAL_COOLDOWN_FRAMES)  — guards against tracker-ID-churn where the
                                            same two fighters get new IDs mid-fight and
                                            bypass the per-pair cooldown.
    """

    def __init__(self):
        self._cooldown: dict[tuple[int, int], int] = {}   # pair → frames remaining
        self._global_cooldown: int = 0                    # global frame counter
        self.events: list[ImpactEvent] = []

    def process(
        self,
        contacts: list[ContactFeatures],
        frame_idx: int,
        fps: float,
    ) -> list[ImpactEvent]:
        """Filter contacts by cooldown, create ImpactEvent records, return new events."""
        new_events: list[ImpactEvent] = []

        # Tick down cooldowns
        self._global_cooldown = max(0, self._global_cooldown - 1)
        expired = [k for k, v in self._cooldown.items() if v <= 0]
        for k in expired:
            del self._cooldown[k]
        for k in self._cooldown:
            self._cooldown[k] -= 1

        for feat in contacts:
            # Global gate: enforces minimum gap between any two impacts
            if self._global_cooldown > 0:
                continue

            pair_key = (feat.aggressor_id, feat.receiver_id)
            rev_key  = (feat.receiver_id, feat.aggressor_id)
            if pair_key in self._cooldown or rev_key in self._cooldown:
                continue

            event = ImpactEvent(
                frame=frame_idx,
                time_sec=frame_idx / max(fps, 1e-6),
                aggressor_id=feat.aggressor_id,
                receiver_id=feat.receiver_id,
                contact_point=feat.contact_point,
                contact_region=feat.contact_region,
                striking_limb=feat.striking_limb,
                probability=feat.contact_probability,
                velocity=feat.velocity,
                impact_type=feat.impact_type,
            )
            new_events.append(event)
            self.events.append(event)
            self._cooldown[pair_key] = IMPACT_COOLDOWN_FRAMES
            self._global_cooldown    = GLOBAL_COOLDOWN_FRAMES

        return new_events

    def summary(self) -> dict:
        head  = sum(1 for e in self.events if e.contact_region == "head")
        torso = sum(1 for e in self.events if e.contact_region == "torso")
        return {
            "total_impacts": len(self.events),
            "head_impacts":  head,
            "torso_impacts": torso,
            "events":        self.events,
        }
