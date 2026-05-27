"""
SAM3D-style renderer using the real SMPL body mesh (6890 vertices, 13776 faces).

Loads basicmodel_m_lbs_10_207_0_v1.0.0.pkl from the checkpoints directory,
partitions vertices by LBS weights into body regions, colours contact regions
in gold and the rest in SAM3D-blue, then renders with pyrender.

Matches the paper figures exactly: smooth shaded SMPL mesh on dark background.
"""
import io
import pickle
import numpy as np
import cv2
import trimesh
import pyrender

# ── Colours ───────────────────────────────────────────────────────────────────
_BODY_RGBA    = np.array([0.24, 0.47, 1.00, 1.0], np.float32)   # SAM3D blue
_CONTACT_RGBA = np.array([1.00, 0.84, 0.00, 1.0], np.float32)   # Pi-HOC gold
_BG           = [0.055, 0.055, 0.07, 1.0]

# ── SMPL joint index → coarse body region ────────────────────────────────────
_JOINT_REGION = {
    0:  "torso",   # Pelvis
    1:  "l_leg",   # L-Hip
    2:  "r_leg",   # R-Hip
    3:  "torso",   # Spine1
    4:  "l_leg",   # L-Knee
    5:  "r_leg",   # R-Knee
    6:  "torso",   # Spine2
    7:  "l_leg",   # L-Ankle
    8:  "r_leg",   # R-Ankle
    9:  "torso",   # Spine3
    10: "l_leg",   # L-Foot
    11: "r_leg",   # R-Foot
    12: "head",    # Neck
    13: "torso",   # L-Collar
    14: "torso",   # R-Collar
    15: "head",    # Head
    16: "l_arm",   # L-Shoulder
    17: "r_arm",   # R-Shoulder
    18: "l_arm",   # L-Elbow
    19: "r_arm",   # R-Elbow
    20: "l_arm",   # L-Wrist
    21: "r_arm",   # R-Wrist
    22: "l_arm",   # L-Hand
    23: "r_arm",   # R-Hand
}

# Fine-grained aliases (from body_contact_viz) → coarse region
_ALIAS = {
    "l_uarm": "l_arm", "l_farm": "l_arm",
    "r_uarm": "r_arm", "r_farm": "r_arm",
    "l_uleg": "l_leg", "l_lleg": "l_leg",
    "r_uleg": "r_leg", "r_lleg": "r_leg",
}


# ── SMPL loader (bypasses broken chumpy on Python 3.11+) ─────────────────────

class _AnyStub:
    """Absorbs chumpy objects; extracts the raw numpy array from pickle state."""
    def __init__(self, *a, **kw):
        self._data = None
    def __setstate__(self, state):
        if isinstance(state, dict) and "x" in state:
            self._data = np.array(state["x"])
    def __array__(self, dtype=None, copy=None):
        return np.array(self._data) if self._data is not None else np.zeros(0)


class _SMPLUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "chumpy" in module:
            return _AnyStub
        return super().find_class(module, name)


def _load_smpl(path: str) -> dict:
    with open(path, "rb") as fh:
        raw = fh.read()
    return _SMPLUnpickler(io.BytesIO(raw), encoding="latin1").load()


# ── Module-level cached SMPL data ────────────────────────────────────────────

_smpl_verts:         np.ndarray | None = None   # (6890, 3)
_smpl_faces:         np.ndarray | None = None   # (13776, 3)
_smpl_vert_region:   np.ndarray | None = None   # (6890,) str  per vertex
_smpl_face_region:   np.ndarray | None = None   # (13776,) str per face


def _ensure_loaded(model_path: str):
    global _smpl_verts, _smpl_faces, _smpl_vert_region, _smpl_face_region
    if _smpl_verts is not None:
        return

    m = _load_smpl(model_path)

    verts   = np.array(m["v_template"],  dtype=np.float32)   # (6890, 3)
    faces   = np.array(m["f"],           dtype=np.int32)      # (13776, 3)
    weights = np.array(m["weights"],     dtype=np.float32)    # (6890, 24)

    # --- centre & normalise so body height ≈ 2 units -------------------------
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    centre = np.array([0.0, (y_min + y_max) / 2.0, 0.0], np.float32)
    scale  = 2.0 / max(y_max - y_min, 1e-6)
    verts  = (verts - centre) * scale    # Y in [-1, +1]

    # --- per-vertex primary joint → region -----------------------------------
    primary_joint  = np.argmax(weights, axis=1)              # (6890,)
    vert_region_list = [_JOINT_REGION[int(j)] for j in primary_joint]
    vert_region    = np.array(vert_region_list)

    # --- per-face region (majority vote) -------------------------------------
    from collections import Counter
    face_region = []
    for tri in faces:
        regions = [vert_region[v] for v in tri]
        face_region.append(Counter(regions).most_common(1)[0][0])
    face_region = np.array(face_region)

    _smpl_verts       = verts
    _smpl_faces       = faces
    _smpl_vert_region = vert_region
    _smpl_face_region = np.array(face_region)


# ── Pyrender helpers ──────────────────────────────────────────────────────────

def _look_at(origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    """4×4 pose where camera sits at `origin` looking at `target` (pyrender: -Z forward)."""
    fwd = target - origin
    fwd /= np.linalg.norm(fwd) + 1e-8
    up  = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(fwd, up)) > 0.98:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up);   right /= np.linalg.norm(right)
    up2   = np.cross(right, fwd)
    M = np.eye(4, dtype=np.float64)
    M[:3, 0] =  right
    M[:3, 1] =  up2
    M[:3, 2] = -fwd      # camera looks down -Z in pyrender
    M[:3, 3] =  origin
    return M


def _mat(rgba: np.ndarray) -> pyrender.MetallicRoughnessMaterial:
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=rgba.tolist(),
        metallicFactor=0.04,
        roughnessFactor=0.55,
        alphaMode="OPAQUE",
    )


# ── Public API ────────────────────────────────────────────────────────────────

SMPL_PATH = "checkpoints/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"


def render_smpl_body(
    contact_regions: list[str],
    width:    int   = 340,
    height:   int   = 420,
    title:    str   = "",
    azim_deg: float = -20.0,
    elev_deg: float =  8.0,
) -> np.ndarray:
    """
    Render the real SMPL body mesh.
    Contact regions are highlighted in gold; the rest is SAM3D-blue.

    contact_regions accepts both coarse ("torso", "head", "l_arm", "r_arm",
    "l_leg", "r_leg") and fine names ("l_uarm", "r_farm", …).
    """
    _ensure_loaded(SMPL_PATH)

    # Resolve fine-grained aliases
    coarse_hits: set[str] = set()
    for r in contact_regions:
        coarse_hits.add(_ALIAS.get(r, r))

    # Build one sub-mesh per region colour ─────────────────────────────────
    scene = pyrender.Scene(bg_color=_BG, ambient_light=[0.18, 0.18, 0.20])
    all_regions = set(_JOINT_REGION.values())

    for region in all_regions:
        mask  = _smpl_face_region == region
        if not mask.any():
            continue
        sub_faces = _smpl_faces[mask]
        used_verts, inv = np.unique(sub_faces, return_inverse=True)
        sub_verts  = _smpl_verts[used_verts]
        sub_faces2 = inv.reshape(-1, 3)

        mesh_tm = trimesh.Trimesh(
            vertices=sub_verts.astype(np.float64),
            faces=sub_faces2,
            process=False,
        )
        rgba = _CONTACT_RGBA if region in coarse_hits else _BODY_RGBA
        scene.add(pyrender.Mesh.from_trimesh(mesh_tm, material=_mat(rgba), smooth=True))

    # Lighting ─────────────────────────────────────────────────────────────
    for intensity, direction in [
        (5.0,  np.array([-0.4,  0.7,  1.0])),   # key: front-top-left
        (2.5,  np.array([ 1.0,  0.3,  0.4])),   # fill: right
        (1.8,  np.array([ 0.1, -0.4, -1.0])),   # rim: back-bottom
    ]:
        d = direction / np.linalg.norm(direction)
        scene.add(
            pyrender.DirectionalLight(color=[1, 1, 1], intensity=intensity),
            pose=_look_at(d * 4, np.zeros(3)),
        )

    # Camera ───────────────────────────────────────────────────────────────
    az  = np.radians(azim_deg)
    el  = np.radians(elev_deg)
    cam_dist = 4.5
    cam_pos  = cam_dist * np.array([
        np.cos(el) * np.sin(az),
        np.sin(el),
        np.cos(el) * np.cos(az),
    ])
    cam_target = np.array([0.0, 0.05, 0.0])
    scene.add(
        pyrender.PerspectiveCamera(yfov=np.radians(36), znear=0.01, zfar=20.0),
        pose=_look_at(cam_pos, cam_target),
    )

    # Render ───────────────────────────────────────────────────────────────
    renderer = pyrender.OffscreenRenderer(width, height)
    colour, _ = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
    renderer.delete()

    img = cv2.cvtColor(colour, cv2.COLOR_RGB2BGR)
    if title:
        cv2.putText(img, title, (8, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (185, 185, 185), 1, cv2.LINE_AA)
    return img


def render_smpl_pair(
    contact_regions_a: list[str],
    contact_regions_b: list[str],
    label_a: str = "",
    label_b: str = "",
    width:  int = 340,
    height: int = 420,
) -> np.ndarray:
    """Side-by-side SMPL bodies: aggressor (left) + receiver (right)."""
    img_a = render_smpl_body(contact_regions_a, width, height, label_a)
    img_b = render_smpl_body(contact_regions_b, width, height, label_b)
    gap   = np.full((height, 8, 3), 12, dtype=np.uint8)
    return np.hstack([img_a, gap, img_b])
