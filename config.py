"""
SAM3D Boxing Impact Detection — Configuration
Pi-HOC (Pairwise 3D Human-Object Contact Estimation) inspired parameters.
Reference: Pi-HOC paper — all hyperparameters preserved from paper + adapted for boxing.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

# ─── Model Config ────────────────────────────────────────────────────────────
YOLO_MODEL = "yolov8n-pose.pt"       # Lightweight pose model; auto-downloads
SAM_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"             # vit_b (~375MB), vit_l (~1.2GB), vit_h (~2.5GB)
SAM_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)

# ─── Detection Config ─────────────────────────────────────────────────────────
YOLO_CONF_THRESHOLD = 0.45           # Person confidence threshold
YOLO_PERSON_CLASS = 0                # COCO class 0 = person
MIN_PERSON_AREA_RATIO = 0.020        # Min fraction of frame area (filters crowd/corner-men)
MAX_PERSON_POOL = 5                  # Detect at most this many before fighter selection

# ─── Pi-HOC Pair Formation (Sec 3.2 of paper) ────────────────────────────────
IOI_THRESHOLD = 0.0                  # γ: IoU threshold for pair formation (paper uses 0)
CONTACT_THRESHOLD = 0.62             # δ: contact presence score threshold (raised from 0.58)

# ─── Impact Detection Thresholds ─────────────────────────────────────────────
# Proximity (pixels) between wrist and opponent body to count as contact
PROXIMITY_THRESHOLD = 60             # pixels — tighter gate (was 70)
VELOCITY_THRESHOLD = 12              # px/frame directed vel that saturates score to 1.0
IMPACT_COOLDOWN_FRAMES = 30          # min frames between same-pair impacts (~1s at 30fps)
GLOBAL_COOLDOWN_FRAMES = 20          # min frames between ANY two impacts regardless of pair
                                     # guards against tracker-ID-churn inflating counts

# Arm extension gate: wrist-to-shoulder / full-arm-length
# Below this value the arm is bent (guard / retraction) — cannot be a punch
ARM_EXTENSION_MIN = 0.68             # raised from 0.62 — require more extension

# Directional velocity gate: minimum velocity component toward opponent (px/frame)
# Filters lateral swings, retracting arms, and tracker-jitter spikes
MIN_DIRECTED_VELOCITY = 9.0          # raised from 6.0

# Hard cap on raw wrist displacement per frame to discard tracker-ID-switch spikes
MAX_WRIST_VELOCITY = 60.0

# SMPL 3D video: frames to hold gold highlight after an impact fires
SMPL_HOLD_FRAMES = 45                # ~1.5 s at 30 fps (processed every 2 frames)

# Four-component scoring weights (must sum to 1.0)
W_PROXIMITY    = 0.35               # spatial proximity score
W_DIRECTED_VEL = 0.40               # velocity component directed toward opponent
W_EXTENSION    = 0.15               # arm extension ratio score
W_CONFIDENCE   = 0.10               # keypoint confidence product

# ─── Pi-HOC Loss Weights (Sec 3 of paper) — kept for documentation ────────────
LAMBDA_2D_FOCAL = 4.0
LAMBDA_DICE = 1.0
LAMBDA_3D_FOCAL = 4.0
LAMBDA_SPARSITY = 0.01
LAMBDA_CP = 1.0

# ─── Processing Config ────────────────────────────────────────────────────────
PROCESS_EVERY_N_FRAMES = 2           # stride — every 2nd frame processed
MAX_FIGHTERS = 2                     # track exactly 2 fighters — referee & corner excluded
OPTICAL_FLOW_WINSIZE = 15            # Lucas-Kanade optical flow window size

# ─── COCO Pose Keypoint Indices ───────────────────────────────────────────────
KP = {
    "nose": 0,
    "left_eye": 1,   "right_eye": 2,
    "left_ear": 3,   "right_ear": 4,
    "left_shoulder": 5,  "right_shoulder": 6,
    "left_elbow": 7,     "right_elbow": 8,
    "left_wrist": 9,     "right_wrist": 10,
    "left_hip": 11,      "right_hip": 12,
    "left_knee": 13,     "right_knee": 14,
    "left_ankle": 15,    "right_ankle": 16,
}

# Body region groups (for contact localization)
HEAD_KPS = [KP["nose"], KP["left_eye"], KP["right_eye"], KP["left_ear"], KP["right_ear"]]
SHOULDER_KPS = [KP["left_shoulder"], KP["right_shoulder"]]
TORSO_KPS = [KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]]
STRIKING_KPS = [KP["left_wrist"], KP["right_wrist"]]   # elbows removed — guard-position false positives

STRIKING_KPS_INFO = [
    (KP["left_wrist"],  "left_jab"),
    (KP["right_wrist"], "right_cross"),
]

# ─── Visualisation ────────────────────────────────────────────────────────────
VIZ_COLORS = {
    "fighter_1":    (255, 100, 0),    # Blue-ish (BGR)
    "fighter_2":    (0, 100, 255),    # Orange-ish (BGR)
    "referee":      (200, 200, 200),  # Grey
    "impact_flash": (0, 0, 255),      # Red flash on impact
    "skeleton":     (0, 255, 180),    # Green-ish skeleton
    "contact_pt":   (0, 255, 255),    # Yellow contact point
    "mask_f1":      (255, 80, 0),
    "mask_f2":      (0, 80, 255),
}
IMPACT_FLASH_DURATION = 6            # frames to show impact flash

# ─── Pre-extracted Keypoint Paths ─────────────────────────────────────────────
_DATA = r"/home/jake/Downloads/for_impact_detection_experiment_2"
KEYPOINTS_2D_PATH = os.path.join(_DATA, "2d_points.json")
KEYPOINTS_3D_PATH = os.path.join(_DATA, "3d_points.json")
ACTIONS_PATH = os.path.join(_DATA, "full_results.json")
_LEGACY_ACTIONS_PATH = (
    r"C:\Users\XRIG\Downloads\for_impact_detection_experiment_2 (1)"
    r"\for_impact_detection_experiment_2\full_results.json"
)

# ─── Impact Detection from Pre-extracted Keypoints ────────────────────────────
IMPACT_SCORE_THRESHOLD = 0.45        # minimum score to classify as landed impact
WINDOW_PAD_FRAMES      = 5          # extend action window by ±N frames for analysis

# Five-gate scoring weights (must sum to 1.0)
DECEL_RATIO_WEIGHT   = 0.30         # wrist deceleration signature
JERK_WEIGHT          = 0.25         # 3D jerk (rate of acceleration change)
EXTENSION_WEIGHT     = 0.15         # arm extension pattern
DEPTH_WEIGHT         = 0.15         # 3D depth convergence
CONFIDENCE_WEIGHT    = 0.15         # ASFormer action confidence boost

# ─── 70-Joint Skeleton Indices (COCO body subset: joints 0-16) ───────────────
KP70_NOSE            = 0
KP70_LEFT_EYE        = 1
KP70_RIGHT_EYE       = 2
KP70_LEFT_EAR        = 3
KP70_RIGHT_EAR       = 4
KP70_LEFT_SHOULDER   = 5
KP70_RIGHT_SHOULDER  = 6
KP70_LEFT_ELBOW      = 7
KP70_RIGHT_ELBOW     = 8
KP70_LEFT_WRIST      = 9
KP70_RIGHT_WRIST     = 10
KP70_LEFT_HIP        = 11
KP70_RIGHT_HIP       = 12
KP70_LEFT_KNEE       = 13
KP70_RIGHT_KNEE      = 14
KP70_LEFT_ANKLE      = 15
KP70_RIGHT_ANKLE     = 16

# Action type → striking wrist index mapping (orthodox stance)
ACTION_HAND_MAP = {
    "jab":                   KP70_LEFT_WRIST,
    "cross":                 KP70_RIGHT_WRIST,
    "hook_left":             KP70_LEFT_WRIST,
    "hook_right":            KP70_RIGHT_WRIST,
    "uppercut_left":         KP70_LEFT_WRIST,
    "uppercut_right":        KP70_RIGHT_WRIST,
    "roundhouse_kick_right": KP70_RIGHT_WRIST,  # closest arm proxy for kicks
    "roundhouse_kick_left":  KP70_LEFT_WRIST,
}

# Action type → corresponding elbow/shoulder for extension analysis
ACTION_ARM_CHAIN = {
    KP70_LEFT_WRIST:  (KP70_LEFT_ELBOW,  KP70_LEFT_SHOULDER),
    KP70_RIGHT_WRIST: (KP70_RIGHT_ELBOW, KP70_RIGHT_SHOULDER),
}
