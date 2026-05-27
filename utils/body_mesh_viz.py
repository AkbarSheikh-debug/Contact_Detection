"""
SAM3D-style 3-D body mesh renderer.

Builds a volumetric T-pose body from trimesh capsule/sphere primitives and renders
it with pyrender using proper PBR materials and three-point lighting — producing
the smooth blue/gold mesh appearance from the SAM3D / Pi-HOC paper figures.

No SMPL model files required: body is a parametric capsule skeleton.
"""
import numpy as np
import cv2
import trimesh

import pyrender

# ── Colour palette ────────────────────────────────────────────────────────────
_BODY_COLOR    = np.array([0.24, 0.47, 1.00, 1.0], dtype=np.float32)   # SAM3D blue
_CONTACT_COLOR = np.array([1.00, 0.85, 0.00, 1.0], dtype=np.float32)   # Pi-HOC gold
_BG_COLOR      = np.array([0.06, 0.06, 0.08, 1.0], dtype=np.float32)

# ── Canonical T-pose joint positions (Y-up, unit = half body height) ─────────
J = {
    "head":      np.array([ 0.00,  1.12,  0.02]),
    "neck":      np.array([ 0.00,  0.85,  0.00]),
    "l_sho":     np.array([-0.24,  0.72,  0.00]),
    "r_sho":     np.array([ 0.24,  0.72,  0.00]),
    "l_elbow":   np.array([-0.58,  0.72,  0.00]),
    "r_elbow":   np.array([ 0.58,  0.72,  0.00]),
    "l_wrist":   np.array([-0.90,  0.72,  0.00]),
    "r_wrist":   np.array([ 0.90,  0.72,  0.00]),
    "l_hip":     np.array([-0.14,  0.20,  0.00]),
    "r_hip":     np.array([ 0.14,  0.20,  0.00]),
    "l_knee":    np.array([-0.17, -0.38,  0.00]),
    "r_knee":    np.array([ 0.17, -0.38,  0.00]),
    "l_ankle":   np.array([-0.18, -0.96,  0.00]),
    "r_ankle":   np.array([ 0.18, -0.96,  0.00]),
    "mid_torso": np.array([ 0.00,  0.46,  0.00]),
}

# ── Body segments: (start_joint, end_joint, radius, contact_region) ──────────
_SEGMENTS = [
    # Torso (wider, handled separately as tapered box)
    # Arms
    ("l_sho",   "l_elbow",  0.065, "l_arm"),
    ("l_elbow", "l_wrist",  0.055, "l_arm"),
    ("r_sho",   "r_elbow",  0.065, "r_arm"),
    ("r_elbow", "r_wrist",  0.055, "r_arm"),
    # Legs
    ("l_hip",   "l_knee",   0.080, "l_leg"),
    ("l_knee",  "l_ankle",  0.065, "l_leg"),
    ("r_hip",   "r_knee",   0.080, "r_leg"),
    ("r_knee",  "r_ankle",  0.065, "r_leg"),
    # Neck
    ("neck",    "head",     0.050, "head"),
]

# Torso is a tapered ellipsoid (built separately)
# Contact region → set of segment regions + special pieces
# Accepts both coarse names ("r_arm") and fine names from body_contact_viz ("r_uarm","r_farm")
_CONTACT_REGIONS_MAP = {
    "head":   {"head"},
    "torso":  {"torso"},
    "l_arm":  {"l_arm"},
    "r_arm":  {"r_arm"},
    "l_leg":  {"l_leg"},
    "r_leg":  {"r_leg"},
    # Fine-grained aliases from the 2D body diagram
    "l_uarm": {"l_arm"},
    "l_farm": {"l_arm"},
    "r_uarm": {"r_arm"},
    "r_farm": {"r_arm"},
    "l_uleg": {"l_leg"},
    "l_lleg": {"l_leg"},
    "r_uleg": {"r_leg"},
    "r_lleg": {"r_leg"},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _capsule_between(p0: np.ndarray, p1: np.ndarray, radius: float) -> trimesh.Trimesh:
    """Create a capsule mesh aligned from p0 to p1."""
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return trimesh.creation.uv_sphere(radius=radius)

    cap = trimesh.creation.capsule(height=length, radius=radius, count=[8, 12])

    # Default capsule is along Z; rotate to align with axis
    z = np.array([0.0, 0.0, 1.0])
    d = axis / length
    angle = np.arccos(np.clip(np.dot(z, d), -1.0, 1.0))
    if angle > 1e-6:
        rot_axis = np.cross(z, d)
        norm     = np.linalg.norm(rot_axis)
        if norm > 1e-6:
            rot_axis /= norm
            R = trimesh.transformations.rotation_matrix(angle, rot_axis)
            cap.apply_transform(R)

    # Translate to midpoint
    mid = (p0 + p1) / 2.0
    cap.apply_translation(mid)
    return cap


def _torso_mesh() -> trimesh.Trimesh:
    """Build the torso as a scaled ellipsoid."""
    sph = trimesh.creation.uv_sphere(radius=1.0, count=[16, 16])
    # Scale: wide at shoulders, narrower at hips, front-back compressed
    sph.vertices[:, 0] *= 0.24   # X (left-right)
    sph.vertices[:, 1] *= 0.35   # Y (up-down)
    sph.vertices[:, 2] *= 0.14   # Z (front-back)
    mid = (J["neck"] + J["l_hip"] + J["r_hip"]) / 3.0
    sph.apply_translation(mid + np.array([0, 0.03, 0]))
    return sph


def _material(color: np.ndarray) -> pyrender.MetallicRoughnessMaterial:
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=color.tolist(),
        metallicFactor=0.05,
        roughnessFactor=0.6,
        alphaMode="OPAQUE",
    )


# ── Main render function ──────────────────────────────────────────────────────

def render_body_mesh(
    contact_regions: list[str],
    width:   int   = 320,
    height:  int   = 380,
    title:   str   = "",
    azim_deg: float = -30.0,
    elev_deg: float = 10.0,
) -> np.ndarray:
    """
    Render a SAM3D-style blue body mesh with contact regions in gold.

    Parameters
    ----------
    contact_regions : list of region names in {"head","torso","l_arm","r_arm","l_leg","r_leg"}
    width, height   : output image size
    title           : label drawn on the image (empty = no label)
    azim_deg, elev_deg : camera rotation around body

    Returns
    -------
    BGR numpy array of shape (height, width, 3)
    """
    # Which segment-regions are hit
    hit_regions: set[str] = set()
    for cr in contact_regions:
        hit_regions |= _CONTACT_REGIONS_MAP.get(cr, set())

    scene = pyrender.Scene(
        bg_color=_BG_COLOR.tolist(),
        ambient_light=[0.20, 0.20, 0.22],
    )

    def _add(mesh_part: trimesh.Trimesh, region: str):
        col = _CONTACT_COLOR if region in hit_regions else _BODY_COLOR
        mat = _material(col)
        scene.add(pyrender.Mesh.from_trimesh(mesh_part, material=mat, smooth=True))

    # Torso
    _add(_torso_mesh(), "torso")

    # Head sphere
    head_sph = trimesh.creation.uv_sphere(radius=0.16, count=[16, 16])
    head_sph.apply_translation(J["head"])
    _add(head_sph, "head")

    # Arm / leg / neck capsules
    for j0, j1, r, region in _SEGMENTS:
        cap = _capsule_between(J[j0], J[j1], r)
        _add(cap, region)

    # Shoulder spheres (small joints)
    for key, region in [("l_sho", "l_arm"), ("r_sho", "r_arm"),
                         ("l_hip", "l_leg"), ("r_hip", "r_leg")]:
        sph = trimesh.creation.uv_sphere(radius=0.068, count=[10, 10])
        sph.apply_translation(J[key])
        _add(sph, region)

    # ── Lighting ──────────────────────────────────────────────────────────────
    # Key light (front-top-left), fill (right), rim (back-top)
    for intensity, direction in [
        (4.5, np.array([-0.5,  0.6,  1.0])),
        (2.0, np.array([ 1.0,  0.2,  0.5])),
        (1.5, np.array([ 0.2, -0.3, -1.0])),
    ]:
        d = direction / np.linalg.norm(direction)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        # Build a pose matrix that points toward -d
        pose = _look_at_pose(origin=d * 3, target=np.zeros(3))
        scene.add(light, pose=pose)

    # ── Camera ────────────────────────────────────────────────────────────────
    cam_dist  = 4.2          # pulled back so full body fits in frame
    az   = np.radians(azim_deg)
    el   = np.radians(elev_deg)
    cam_pos = cam_dist * np.array([
        np.cos(el) * np.sin(az),
        np.sin(el),
        np.cos(el) * np.cos(az),
    ])
    cam_target = np.array([0.0, 0.10, 0.0])   # look slightly above origin (body center)
    cam_pose   = _look_at_pose(cam_pos, cam_target)

    cam = pyrender.PerspectiveCamera(yfov=np.radians(38), znear=0.01, zfar=20.0)
    scene.add(cam, pose=cam_pose)

    # ── Render ────────────────────────────────────────────────────────────────
    renderer = pyrender.OffscreenRenderer(width, height)
    color_rgb, _ = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
    renderer.delete()

    img_bgr = cv2.cvtColor(color_rgb[:, :, :3], cv2.COLOR_RGB2BGR)

    if title:
        cv2.putText(img_bgr, title, (8, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)
    return img_bgr


def render_pair_mesh(
    contact_regions_a: list[str],
    contact_regions_b: list[str],
    label_a: str = "",
    label_b: str = "",
    width:  int = 320,
    height: int = 380,
) -> np.ndarray:
    """Side-by-side mesh renders for aggressor (left) and receiver (right)."""
    img_a = render_body_mesh(contact_regions_a, width, height, label_a)
    img_b = render_body_mesh(contact_regions_b, width, height, label_b)
    gap   = np.full((height, 8, 3), 14, dtype=np.uint8)
    return np.hstack([img_a, gap, img_b])


# ── Utility: look-at pose matrix (pyrender convention: camera looks down -Z) ──

def _look_at_pose(origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Returns a 4×4 pose matrix for a pyrender camera/light placed at `origin`
    looking toward `target`.  In pyrender the camera/light looks down its -Z axis.
    """
    # forward = origin → target (this becomes -Z in camera space)
    fwd = target - origin
    fwd /= (np.linalg.norm(fwd) + 1e-8)

    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(fwd, up)) > 0.98:
        up = np.array([0.0, 0.0, 1.0])

    right = np.cross(fwd, up);   right /= np.linalg.norm(right)
    up2   = np.cross(right, fwd)

    M = np.eye(4)
    M[:3, 0] =  right          # X  = right
    M[:3, 1] =  up2            # Y  = up
    M[:3, 2] = -fwd            # Z  = BACK (camera looks down -Z)
    M[:3, 3] =  origin
    return M
