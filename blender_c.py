import bpy
import math
import os
import mathutils
import json
import random
import sys
import time

# ==============================
# GPU-OPTIMIZED V2: Max GPU, minimal CPU
# - Frame-first rendering (1 frame_set per frame)
# - NO raycasts (skip/caged rules handle occlusion)
# - Sampled bbox (every 20th frame instead of all 900)
# - CPU cooldown throttle between frames
# ==============================
print("🚀 GPU-Optimized V2: No raycasts, sampled bbox, CPU throttle")

# ==============================
# CONFIG
# ==============================
person = bpy.data.objects["Armature"]

output_dir = r"C:\Users\XRIG\Desktop\New_Blender_Sprints\Synthetic_dataset\multi_test_3"

heights = [0.5, 2, 4, 6, 8, 10]

height_angle_ranges = {
    0.5: (0, 4),
    2:   (4, 8),
    4:   (8, 12),
    6:   (12, 16),
    8:   (16, 20),
    10:  (20, 24),
}

total_angles = 24

start_frame = 1
end_frame = 730

selected_frames = list(range(start_frame, end_frame + 1, 2))

# Sample every 20th frame for bbox computation (90 instead of 900)
bbox_sample_frames = list(range(start_frame, end_frame + 1, 20))

light_powers = [800]

light_colors = [
    (1.0, 1.0, 1.0, 1),
]

distance_pairs = [
    (3, 4),
    (5, 6),
    (7, 9),
    (10, 13),
    (14, 17),
    (18, 22),
    (25, 30),
]

# CPU cooldown: pause this many seconds every N frames to let CPU cool
CPU_COOLDOWN_INTERVAL = 100   # every 100 frames
CPU_COOLDOWN_SECONDS = 4.0    # pause 2 seconds

# ==============================
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'

scene.eevee.taa_render_samples = 32
scene.render.use_motion_blur = False
scene.render.use_compositing = False
scene.render.use_sequencer = False

eevee = scene.eevee
for attr in ['use_ssr', 'use_bloom', 'use_volumetric_lights', 'use_volumetric_shadows']:
    if hasattr(eevee, attr):
        setattr(eevee, attr, False)

for attr, val in [('shadow_cube_size', '512'), ('shadow_cascade_size', '512')]:
    if hasattr(eevee, attr):
        setattr(eevee, attr, val)

for obj in bpy.data.objects:
    if obj.type == 'LIGHT':
        obj.data.use_shadow = False

for attr in ['use_shadows', 'use_shadow_high_bitdepth', 'use_soft_shadows']:
    if hasattr(eevee, attr):
        setattr(eevee, attr, False)

try:
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        device.use = 'RTX' in device.name or 'NVIDIA' in device.name
except Exception:
    pass

# ==============================
# QUALITY SETTINGS
# ==============================
scene.render.image_settings.file_format = 'JPEG'
scene.render.image_settings.quality = 65
scene.render.resolution_x = 960
scene.render.resolution_y = 540
scene.render.resolution_percentage = 100

# ==============================
# ACTIONS
# ==============================
actions = [
    {"action": "cross_right", "start": 43, "end": 63},
    {"action": "jab_left", "start": 83, "end": 98},
    {"action": "uppercut_right", "start": 98, "end": 117},
    {"action": "elbow_left", "start": 251, "end": 278},
    {"action": "front_kick_right", "start": 437, "end": 462},
    {"action": "low_kick_left", "start": 596, "end": 621},
]

os.makedirs(output_dir, exist_ok=True)

# ==============================
# ARMATURE POSITIONS IN THE RING
# ==============================
ring_offsets = [
    (0.0, 0.0),
    (-1.4, 1.4),
]

original_armature_loc = person.location.copy()

# ==============================
# JSON
# ==============================
json_path = os.path.join(output_dir, "frame_labels.json")

if not os.path.exists(json_path):
    frame_labels = {}
    for frame in selected_frames:
        frame_labels[frame] = "background"
    for action in actions:
        for f in selected_frames:
            if action["start"] <= f <= action["end"]:
                frame_labels[f] = action["action"]
    with open(json_path, "w") as f:
        json.dump(frame_labels, f, indent=4)

# ==============================
# BOUNDING BOX (sampled — every 20th frame)
# ==============================
def get_bbox_world(obj):
    return [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]

min_corner = mathutils.Vector((1e9, 1e9, 1e9))
max_corner = mathutils.Vector((-1e9, -1e9, -1e9))

print(f"📐 Computing bbox from {len(bbox_sample_frames)} sampled frames (instead of {len(selected_frames)})")

for frame in bbox_sample_frames:
    bpy.context.scene.frame_set(frame)
    bbox = get_bbox_world(person)
    for v in bbox:
        min_corner.x = min(min_corner.x, v.x)
        min_corner.y = min(min_corner.y, v.y)
        min_corner.z = min(min_corner.z, v.z)
        max_corner.x = max(max_corner.x, v.x)
        max_corner.y = max(max_corner.y, v.y)
        max_corner.z = max(max_corner.z, v.z)

body_height = max_corner.z - min_corner.z
base_bottom_z = min_corner.z
base_top_z = max_corner.z

print(f"📏 Body height: {body_height:.2f} | Ground: {base_bottom_z:.2f} | Top: {base_top_z:.2f}")

# ==============================
# SETUP
# ==============================
def ensure_camera(name):
    if name in bpy.data.objects:
        return bpy.data.objects[name]
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = name
    return cam

def ensure_light(name):
    if name in bpy.data.objects:
        return bpy.data.objects[name]
    bpy.ops.object.light_add(type='AREA')
    light = bpy.context.active_object
    light.name = name
    return light

def add_track(obj, target_obj):
    obj.constraints.clear()
    c = obj.constraints.new(type='TRACK_TO')
    c.target = target_obj
    c.track_axis = 'TRACK_NEGATIVE_Z'
    c.up_axis = 'UP_Y'

camera_A = ensure_camera("Camera_A")
camera_B = ensure_camera("Camera_B")
light_A = ensure_light("Light_A")
light_B = ensure_light("Light_B")

light_A.data.use_shadow = False
light_B.data.use_shadow = False

if "Track_Target" not in bpy.data.objects:
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    track_target = bpy.context.active_object
    track_target.name = "Track_Target"
else:
    track_target = bpy.data.objects["Track_Target"]

for obj in (camera_A, camera_B, light_A, light_B):
    add_track(obj, track_target)

# ==============================
# HELPERS
# ==============================
def compute_focal_length(cam_location, cur_center, cur_top_z, cur_bottom_z, body_h, margin=1.4):
    cam_pos = mathutils.Vector(cam_location)
    dx = cam_pos.x - cur_center.x
    dy = cam_pos.y - cur_center.y
    horizontal_dist = math.sqrt(dx * dx + dy * dy)
    if horizontal_dist < 0.5:
        horizontal_dist = 0.5
    angle_to_head = math.atan2(cur_top_z - cam_pos.z, horizontal_dist)
    angle_to_feet = math.atan2(cur_bottom_z - cam_pos.z, horizontal_dist)
    total_angle = (angle_to_head - angle_to_feet) * margin
    sensor_h = 36.0 * (scene.render.resolution_y / scene.render.resolution_x)
    half_fov = total_angle / 2.0
    if half_fov < 0.01:
        half_fov = 0.01
    focal = (sensor_h / 2.0) / math.tan(half_fov)
    return max(18.0, min(focal, 85.0))

def get_camera_z(height, distance, cur_bottom_z, cur_eye_z):
    jitter = random.uniform(-0.1, 0.1)
    if height <= 1.4 and distance < body_height:
        return cur_eye_z + jitter
    return cur_bottom_z + height + jitter

# ==============================
# SKIP & CAGED RULES
# ==============================
skip_A = {
    0.5: {3}, 2: {3}, 4: {3}, 6: {3, 5}, 8: {3, 5, 7}, 10: {3, 5, 7},
}
caged_A = {
    0.5: {14, 18}, 2: {14, 18, 25}, 4: {14, 18, 25}, 6: {25}, 8: {25}, 10: set(),
}
skip_B = {
    0.5: set(), 2: set(), 4: set(), 6: set(), 8: {4}, 10: {4},
}
caged_B = {
    0.5: {13, 17, 22, 30}, 2: {13, 17, 22, 30}, 4: {13, 17, 22, 30},
    6: {22, 30}, 8: {30}, 10: set(),
}

def is_caged_A(height, dist):
    return dist in caged_A.get(height, set())

def is_caged_B(height, dist):
    return dist in caged_B.get(height, set())

def is_skipped_A(height, dist):
    return dist in skip_A.get(height, set())

def is_skipped_B(height, dist):
    return dist in skip_B.get(height, set())

def get_folder_A(base, power, pos_idx, height, c_idx, dist, angle_idx):
    if is_skipped_A(height, dist):
        return None
    if is_caged_A(height, dist):
        return os.path.join(base, f"power_{power}", "CameraA",
                            f"pos_{pos_idx}", f"h_{height}", f"color_{c_idx}",
                            "caged", f"dist_{dist}", f"angle_{angle_idx:03d}")
    return os.path.join(base, f"power_{power}", "CameraA",
                        f"pos_{pos_idx}", f"h_{height}", f"color_{c_idx}",
                        f"dist_{dist}", f"angle_{angle_idx:03d}")

def get_folder_B(base, power, pos_idx, height, c_idx, dist, angle_idx):
    if is_skipped_B(height, dist):
        return None
    if is_caged_B(height, dist):
        return os.path.join(base, f"power_{power}", "CameraB",
                            f"pos_{pos_idx}", f"h_{height}", f"color_{c_idx}",
                            "caged", f"dist_{dist}", f"angle_{angle_idx:03d}")
    return os.path.join(base, f"power_{power}", "CameraB",
                        f"pos_{pos_idx}", f"h_{height}", f"color_{c_idx}",
                        f"dist_{dist}", f"angle_{angle_idx:03d}")

# ==============================
# ✅ PRE-COMPUTE ALL RENDER JOBS
# ==============================
def build_render_jobs(pos_idx, center, top_z, bottom_z, eye_level_z, power, render_caged):
    jobs = []

    for height in heights:
        for c_idx, color in enumerate(light_colors):
            for dist_A, dist_B in distance_pairs:

                a_is_caged = is_caged_A(height, dist_A)
                b_is_caged = is_caged_B(height, dist_B)

                process_A = not is_skipped_A(height, dist_A) and (a_is_caged == render_caged)
                process_B = not is_skipped_B(height, dist_B) and (b_is_caged == render_caged)

                if not process_A and not process_B:
                    continue

                ang_start, ang_end = height_angle_ranges[height]

                for angle_idx in range(ang_start, ang_end):

                    folder_A = get_folder_A(output_dir, power, pos_idx, height, c_idx, dist_A, angle_idx) if process_A else None
                    folder_B = get_folder_B(output_dir, power, pos_idx, height, c_idx, dist_B, angle_idx) if process_B else None

                    if folder_A is None and folder_B is None:
                        continue

                    if folder_A is not None:
                        os.makedirs(folder_A, exist_ok=True)
                    if folder_B is not None:
                        os.makedirs(folder_B, exist_ok=True)

                    angle = 2 * math.pi * angle_idx / total_angles

                    cam_z_A = get_camera_z(height, dist_A, bottom_z, eye_level_z)
                    cam_z_B = get_camera_z(height, dist_B, bottom_z, eye_level_z)

                    cam_A_loc = (
                        center.x + dist_A * math.cos(angle),
                        center.y + dist_A * math.sin(angle),
                        cam_z_A
                    )
                    cam_B_loc = (
                        center.x + dist_B * math.cos(angle + math.pi),
                        center.y + dist_B * math.sin(angle + math.pi),
                        cam_z_B
                    )

                    jobs.append({
                        'folder_A': folder_A,
                        'folder_B': folder_B,
                        'cam_A_loc': cam_A_loc,
                        'cam_B_loc': cam_B_loc,
                        'focal_A': compute_focal_length(cam_A_loc, center, top_z, bottom_z, body_height),
                        'focal_B': compute_focal_length(cam_B_loc, center, top_z, bottom_z, body_height),
                        'light_setup': [
                            (dist_A, angle, cam_z_A),
                            (dist_B, angle + math.pi, cam_z_B)
                        ],
                        'color': color,
                        'power': power,
                        'center': center,
                    })

    return jobs


# ==============================
# ✅ FRAME-FIRST RENDER (NO RAYCASTS, WITH CPU COOLDOWN)
# ==============================
def render_frame_first(pass_name, render_caged):
    print(f"\n{'='*50}")
    print(f"🎬 PASS: {pass_name}")
    print(f"{'='*50}")

    for pos_idx, (off_x, off_y) in enumerate(ring_offsets):

        person.location = (
            original_armature_loc.x + off_x,
            original_armature_loc.y + off_y,
            original_armature_loc.z
        )
        bpy.context.view_layer.update()

        print(f"📍 [{pass_name}] Position {pos_idx}: offset ({off_x}, {off_y})")

        # Sampled bbox for this position
        pos_min = mathutils.Vector((1e9, 1e9, 1e9))
        pos_max = mathutils.Vector((-1e9, -1e9, -1e9))

        for frame in bbox_sample_frames:
            bpy.context.scene.frame_set(frame)
            bbox = get_bbox_world(person)
            for v in bbox:
                pos_min.x = min(pos_min.x, v.x)
                pos_min.y = min(pos_min.y, v.y)
                pos_min.z = min(pos_min.z, v.z)
                pos_max.x = max(pos_max.x, v.x)
                pos_max.y = max(pos_max.y, v.y)
                pos_max.z = max(pos_max.z, v.z)

        center = (pos_min + pos_max) / 2
        top_z = pos_max.z
        bottom_z = pos_min.z
        eye_level_z = bottom_z + body_height * 0.90
        adjusted_z = bottom_z + body_height * 0.50

        adjusted_center = center.copy()
        adjusted_center.z = adjusted_z
        track_target.location = adjusted_center

        # Pre-compute ALL camera jobs
        for power in light_powers:
            jobs = build_render_jobs(pos_idx, center, top_z, bottom_z, eye_level_z, power, render_caged)

        print(f"  📋 {len(jobs)} camera setups × {len(selected_frames)} frames")

        if not jobs:
            continue

        render_count = 0

        for f_idx, frame in enumerate(selected_frames):

            # ✅ frame_set() ONCE per frame
            bpy.context.scene.frame_set(frame)

            # Render ALL camera positions for this frame
            for job in jobs:
                folder_A = job['folder_A']
                folder_B = job['folder_B']

                path_A = os.path.join(folder_A, f"frame_{frame:04d}.jpg") if folder_A else None
                path_B = os.path.join(folder_B, f"frame_{frame:04d}.jpg") if folder_B else None

                a_needs = path_A is not None and not os.path.exists(path_A)
                b_needs = path_B is not None and not os.path.exists(path_B)

                if not a_needs and not b_needs:
                    continue

                # Set up lights
                cntr = job['center']
                for light, (r, ang, cz) in zip([light_A, light_B], job['light_setup']):
                    light.location = (
                        cntr.x - r * math.cos(ang),
                        cntr.y - r * math.sin(ang),
                        cz + 1
                    )
                    light.data.energy = job['power']
                    light.data.color = job['color'][:3]

                # ✅ GPU renders — back to back, no raycast between them
                if a_needs:
                    camera_A.location = job['cam_A_loc']
                    camera_A.data.lens = job['focal_A']
                    scene.camera = camera_A
                    scene.render.filepath = path_A
                    bpy.ops.render.render(write_still=True)
                    render_count += 1

                if b_needs:
                    camera_B.location = job['cam_B_loc']
                    camera_B.data.lens = job['focal_B']
                    scene.camera = camera_B
                    scene.render.filepath = path_B
                    bpy.ops.render.render(write_still=True)
                    render_count += 1

            # ✅ CPU COOLDOWN: let CPU breathe every N frames
            if (f_idx + 1) % CPU_COOLDOWN_INTERVAL == 0:
                print(f"  🖼️ Frame {frame} ({f_idx+1}/{len(selected_frames)}) — {render_count} renders — cooling CPU {CPU_COOLDOWN_SECONDS}s...")
                time.sleep(CPU_COOLDOWN_SECONDS)

            # Progress every 50 frames
            elif (f_idx + 1) % 50 == 0:
                print(f"  🖼️ Frame {frame} ({f_idx+1}/{len(selected_frames)}) — {render_count} renders")

        print(f"  ✅ Position {pos_idx} done: {render_count} total renders")


# ==============================
# ✅ PASS 1: ALL NORMAL FOLDERS FIRST
# ==============================
render_frame_first("NORMAL", render_caged=False)

# ==============================
# ✅ PASS 2: ALL CAGED FOLDERS AFTER
# ==============================
render_frame_first("CAGED", render_caged=True)

# ✅ Restore original armature position
person.location = original_armature_loc
bpy.context.view_layer.update()

print("✅ DONE — GPU-optimized V2, 24 unique angles across 6 heights")
