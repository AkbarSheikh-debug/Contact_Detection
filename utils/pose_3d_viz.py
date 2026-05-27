"""
SAM3D-style 3D body pose visualization.
Renders a canonical T-pose body with contact regions highlighted in gold,
matching the "SAM3D before/after refinement" figure style from the paper.

We use a canonical parametric body (not lifted from 2D pose) because the
boxing camera is elevated — lifting the 2D keypoints directly produces a
flat/horizontal pose that misrepresents the body shape.
"""
import io
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Canonical T-pose joint positions (normalised, Y-up) ──────────────────────
# Coordinate system: X = right, Y = up, Z = forward
# Units: 1.0 = half body height

_T_POSE = np.array([
    [ 0.00,  1.00,  0.05],   #  0 nose
    [-0.05,  1.02,  0.04],   #  1 left_eye
    [ 0.05,  1.02,  0.04],   #  2 right_eye
    [-0.12,  0.98,  0.00],   #  3 left_ear
    [ 0.12,  0.98,  0.00],   #  4 right_ear
    [-0.22,  0.72,  0.00],   #  5 left_shoulder
    [ 0.22,  0.72,  0.00],   #  6 right_shoulder
    [-0.55,  0.72,  0.00],   #  7 left_elbow
    [ 0.55,  0.72,  0.00],   #  8 right_elbow
    [-0.88,  0.72,  0.00],   #  9 left_wrist
    [ 0.88,  0.72,  0.00],   # 10 right_wrist
    [-0.14,  0.20,  0.00],   # 11 left_hip
    [ 0.14,  0.20,  0.00],   # 12 right_hip
    [-0.17, -0.35,  0.00],   # 13 left_knee
    [ 0.17, -0.35,  0.00],   # 14 right_knee
    [-0.18, -0.95,  0.00],   # 15 left_ankle
    [ 0.18, -0.95,  0.00],   # 16 right_ankle
], dtype=float)

_BONES = [
    (0, 1),(0, 2),(1, 3),(2, 4),   # head
    (5, 6),                         # shoulders
    (5, 7),(7, 9),                  # left arm
    (6, 8),(8,10),                  # right arm
    (5,11),(6,12),                  # torso sides
    (11,12),                        # hips
    (11,13),(13,15),                # left leg
    (12,14),(14,16),                # right leg
]

# Contact region → which joints to highlight
_REGION_JOINTS = {
    "head":  [0, 1, 2, 3, 4],
    "torso": [5, 6, 11, 12],
    "l_arm": [5, 7, 9],
    "r_arm": [6, 8, 10],
    "l_leg": [11, 13, 15],
    "r_leg": [12, 14, 16],
}

_BODY_BLUE   = "#3C78FF"
_CONTACT_COL = "#FFD700"   # gold for contacted region (matches SAM3D paper)
_BG_COL      = "#111111"


def _cylinder_faces(p0, p1, r=0.04, n=8):
    """Return Poly3DCollection face list for a cylinder from p0 to p1."""
    axis = p1 - p0
    lng  = np.linalg.norm(axis)
    if lng < 1e-6:
        return []
    axis /= lng
    tmp  = np.array([0., 0., 1.]) if abs(axis[2]) < 0.9 else np.array([1., 0., 0.])
    u    = np.cross(axis, tmp);  u /= np.linalg.norm(u)
    v    = np.cross(axis, u)
    θ    = np.linspace(0, 2*np.pi, n, endpoint=False)
    r0   = [p0 + r*(np.cos(a)*u + np.sin(a)*v) for a in θ]
    r1   = [p1 + r*(np.cos(a)*u + np.sin(a)*v) for a in θ]
    return [[r0[i], r0[(i+1)%n], r1[(i+1)%n], r1[i]] for i in range(n)]


def render_3d_body(
    contact_regions: list[str],
    width:  int   = 300,
    height: int   = 340,
    title:  str   = "",
    elev:   float = 8.0,
    azim:   float = -65.0,
) -> np.ndarray:
    """
    Render a canonical upright T-pose body with contact regions in gold.
    contact_regions: subset of ["head","torso","l_arm","r_arm","l_leg","r_leg"].
    Returns BGR numpy array (height × width × 3).
    """
    contacted_joints = {
        j for reg in contact_regions for j in _REGION_JOINTS.get(reg, [])
    }

    fig = plt.figure(figsize=(width/100, height/100), dpi=100, facecolor=_BG_COL)
    ax  = fig.add_subplot(111, projection="3d", facecolor=_BG_COL)

    # Matplotlib 3D is Z-up; our T-pose is Y-up → swap Y↔Z for plotting
    def _p(pt):
        return np.array([pt[0], pt[2], pt[1]])   # (x, z_depth, y_height)

    kp = np.array([_p(pt) for pt in _T_POSE])

    # Bones as cylinders
    for a, b in _BONES:
        is_hit = (a in contacted_joints or b in contacted_joints)
        col    = _CONTACT_COL if is_hit else _BODY_BLUE
        faces  = _cylinder_faces(kp[a], kp[b], r=0.055)
        if faces:
            ax.add_collection3d(
                Poly3DCollection(faces, alpha=0.90, facecolor=col, edgecolor="none")
            )

    # Joint spheres
    for i, pt in enumerate(kp):
        col  = _CONTACT_COL if i in contacted_joints else _BODY_BLUE
        size = 90 if i < 5 else 50
        ax.scatter(pt[0], pt[1], pt[2], c=col, s=size, depthshade=False, zorder=5)

    # Head sphere
    hcol = _CONTACT_COL if any(j < 5 for j in contacted_joints) else _BODY_BLUE
    u = np.linspace(0, 2*np.pi, 18)
    v = np.linspace(0, np.pi, 12)
    r = 0.17
    hd = kp[0]
    ax.plot_surface(
        hd[0] + r*np.outer(np.cos(u), np.sin(v)),
        hd[1] + r*np.outer(np.sin(u), np.sin(v)),
        hd[2] + r*np.outer(np.ones(18), np.cos(v)),
        color=hcol, alpha=0.88, linewidth=0,
    )

    # Styling
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False;  pane.set_edgecolor("none")
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)

    lim = 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)

    if title:
        ax.set_title(title, color="white", fontsize=7, pad=1)

    plt.tight_layout(pad=0.05)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=_BG_COL, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.resize(img, (width, height)) if img is not None else np.zeros((height, width, 3), dtype=np.uint8)


def render_pair_3d(
    contact_regions_a: list[str],
    contact_regions_b: list[str],
    label_a: str = "Aggressor",
    label_b: str = "Receiver",
    width:   int = 300,
    height:  int = 340,
) -> np.ndarray:
    """Side-by-side canonical 3D bodies for an impact pair."""
    img_a = render_3d_body(contact_regions_a, width, height, label_a)
    img_b = render_3d_body(contact_regions_b, width, height, label_b)
    gap   = np.full((height, 8, 3), 20, dtype=np.uint8)
    return np.hstack([img_a, gap, img_b])
