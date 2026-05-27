"""
Pi-HOC style canonical body contact visualization.
Renders T-pose body diagrams with colored contact regions — matching paper Fig 1.
"""
import numpy as np
import cv2

# ── Canvas geometry ───────────────────────────────────────────────────────────
BW, BH = 200, 240          # wider canvas so T-pose arms are visible (w × h)
_SX    = BW / 200.0
_SY    = BH / 220.0

# ── Body region geometry (normalized 0-200 x, 0-220 y, T-pose) ───────────────
# Origin top-left.  Arms extend far left/right, legs go down.
_REGIONS_RAW = {
    "head":   {"type": "circle", "cx": 100, "cy": 14, "r": 13},
    "torso":  {"type": "poly",
               "pts": [(62,28),(138,28),(144,100),(56,100)]},
    # left arm: shoulder→elbow→wrist (horizontal, extending left)
    "l_uarm": {"type": "poly",
               "pts": [(62,30),(62,55),(22,52),(22,32)]},
    "l_farm": {"type": "poly",
               "pts": [(22,32),(22,55),(4,52),(4,34)]},
    # right arm: mirrored
    "r_uarm": {"type": "poly",
               "pts": [(138,30),(178,32),(178,52),(138,55)]},
    "r_farm": {"type": "poly",
               "pts": [(178,32),(196,34),(196,52),(178,52)]},
    # legs
    "l_uleg": {"type": "poly",
               "pts": [(56,100),(96,100),(94,155),(58,155)]},
    "l_lleg": {"type": "poly",
               "pts": [(58,155),(94,155),(92,210),(60,210)]},
    "r_uleg": {"type": "poly",
               "pts": [(104,100),(144,100),(142,155),(106,155)]},
    "r_lleg": {"type": "poly",
               "pts": [(106,155),(142,155),(140,210),(108,210)]},
}

_BASE         = (148, 148, 148)   # uncontacted region colour
_BG           = (14,  14,  14)    # background

_HIT_COLOR = {
    "head":   (0,   215, 255),   # cyan
    "torso":  (190,  45, 255),   # magenta
    "l_uarm": (0,   200,  90),   # green
    "r_uarm": (0,   180,  70),
    "l_farm": (50,  255, 150),
    "r_farm": (30,  235, 120),
    "l_uleg": (255, 150,  30),   # orange
    "r_uleg": (240, 130,  20),
    "l_lleg": (255, 200,  70),   # yellow
    "r_lleg": (240, 185,  50),
}

_STRIKE_COLOR = (100, 190, 255)   # arm that was punching — warm light colour

_RECV_REGION = {"head": ["head"],  "torso": ["torso"]}
_STRIKE_LIMB = {"left_jab":    ["l_uarm", "l_farm"],
                "right_cross": ["r_uarm", "r_farm"]}


# ── Pre-compute pixel masks once at module load ───────────────────────────────
def _build_masks() -> dict[str, np.ndarray]:
    out = {}
    for name, spec in _REGIONS_RAW.items():
        m = np.zeros((BH, BW), dtype=np.uint8)
        if spec["type"] == "circle":
            cx = int(spec["cx"] * _SX)
            cy = int(spec["cy"] * _SY)
            r  = int(spec["r"]  * min(_SX, _SY))
            cv2.circle(m, (cx, cy), r, 255, -1)
        else:
            pts = np.array(
                [(int(x * _SX), int(y * _SY)) for x, y in spec["pts"]],
                dtype=np.int32,
            )
            cv2.fillPoly(m, [pts], 255)
        out[name] = m
    return out

_MASKS = _build_masks()


class BodyContactDiagram:
    """
    Generates Pi-HOC style canonical body contact maps.

    Usage:
        diag = BodyContactDiagram()
        img  = diag.draw(hit_regions=["head", "torso"],
                         strike_regions=["r_uarm", "r_farm"],
                         label="Fighter 0")
    """

    def draw(
        self,
        hit_regions:    list[str] = (),
        strike_regions: list[str] = (),
        label:          str = "",
        hit_counts:     dict[str, int] | None = None,
    ) -> np.ndarray:
        """Return BH×BW×3 BGR body diagram with colored contact regions."""
        img = np.full((BH, BW, 3), _BG, dtype=np.uint8)

        # Grey base body
        for mask in _MASKS.values():
            img[mask > 0] = _BASE

        # Color hit regions
        for region in hit_regions:
            mask = _MASKS.get(region)
            if mask is None:
                continue
            color = _HIT_COLOR.get(region, (0, 200, 255))
            cnt   = (hit_counts or {}).get(region, 1)
            alpha = min(0.92, 0.50 + 0.14 * cnt)
            ov    = img.copy()
            ov[mask > 0] = color
            cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

        # Color strike regions
        for region in strike_regions:
            mask = _MASKS.get(region)
            if mask is None:
                continue
            ov = img.copy()
            ov[mask > 0] = _STRIKE_COLOR
            cv2.addWeighted(ov, 0.60, img, 0.40, 0, img)

        # Region contours for clarity
        for mask in _MASKS.values():
            ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, ctrs, -1, (55, 55, 55), 1)

        if label:
            cv2.putText(img, label, (4, BH - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1)
        return img

    @staticmethod
    def regions_from_event(contact_region: str, striking_limb: str):
        """Return (hit_regions, strike_regions) from ImpactEvent fields."""
        return (_RECV_REGION.get(contact_region, []),
                _STRIKE_LIMB.get(striking_limb, []))
