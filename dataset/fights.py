"""
Fight registry — single source of truth for per-match paths/config used by
dataset/prepare_fight_dataset.py, tools/annotate_clips.py, and
tools/extract_keypoint_dataset.py.

To add a new fight: add an entry to FIGHTS below (copy the lillyella_vs_zoe
block as a template) pointing at that match's video folder + Gladius output
folder, then run:

    python dataset/prepare_fight_dataset.py --fight <name>
    python tools/annotate_clips.py --fight <name>

`identity_marker` is optional: it describes a visual cue (e.g. a fighter's
shorts colour) used to detect CUTIE tracker identity swaps during clinches.
Leave it as None for fights where no such cue is available/needed — the
tracker IDs are then trusted as-is with no swap correction.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = r"C:\Users\XRIG\Downloads\ffmpeg_extracted\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"

FIGHTS = {
    "lillyella_vs_zoe": dict(
        video_folder=(
            r"C:\Users\XRIG\Downloads\drive-download-20260615T202203Z-3-001"
            r"\(BLUE) Lillyella Craw Seaman VS (RED) Zoe Hunte-Smith"
        ),
        gladius_folder=(
            r"C:\Users\XRIG\Desktop\Gladius_Output"
            r"\(BLUE)_Lillyella_Craw_Seaman_VS_(RED)_Zoe_Hunte-Smith"
        ),
        out_base=os.path.join(ROOT, "dataset", "lillyella_vs_zoe"),
        rounds=[3, 4, 5, 8],
        fighter_names={0: "Lillyella (blue)", 1: "Zoe (red)"},
        fighter_short_names={0: "Lillyella", 1: "Zoe"},
        # Zoe's shorts have a bright green dragon trim; Lillyella's are plain
        # black. Checked in the lower (shorts) 55-100% of each fighter's bbox.
        identity_marker={"fighter_id": 1, "color": "green", "region": (0.55, 1.0)},
    ),

    "cameron_vs_liam": dict(
        video_folder=(
            r"C:\Users\XRIG\Downloads\drive-download-20260615T202203Z-3-001"
            r"\(BLUE) Cameron O_Callaghan VS (RED) Liam McElhinney"
        ),
        gladius_folder=(
            r"C:\Users\XRIG\Desktop\Gladius_Output"
            r"\(BLUE)_Cameron_OCallaghan_VS_(RED)_Liam_McElhinney"
        ),
        out_base=os.path.join(ROOT, "dataset", "cameron_vs_liam"),
        # Round1 has no video yet; Round2/4 are each missing one fighter's
        # SAM3D file (clip extraction still works, only the missing fighter's
        # name overlay / keypoint features for that round are unavailable).
        rounds=[2, 3, 4],
        fighter_names={0: "Cameron O'Callaghan (blue)", 1: "Liam McElhinney (red)"},
        fighter_short_names={0: "Cameron", 1: "Liam"},
        identity_marker=None,  # no known visual cue yet for swap detection
    ),

    "jamie_vs_ryan": dict(
        video_folder=(
            r"C:\Users\XRIG\Downloads\drive-download-20260615T202203Z-3-001"
            r"\(BLUE) Jamie Barrett VS (RED) Ryan Frost"
        ),
        gladius_folder=(
            r"C:\Users\XRIG\Desktop\Gladius_Output"
            r"\(BLUE)_Jamie_Barrett_VS_(RED)_Ryan_Frost"
        ),
        out_base=os.path.join(ROOT, "dataset", "jamie_vs_ryan"),
        # Round2/4 have no video yet; Round3 is missing fighter1's SAM3D.
        rounds=[1, 3],
        fighter_names={0: "Jamie Barrett (blue)", 1: "Ryan Frost (red)"},
        fighter_short_names={0: "Jamie", 1: "Ryan"},
        identity_marker=None,
    ),

    # Add new fights here, e.g.:
    # "newfighter_vs_other": dict(
    #     video_folder=r"...",
    #     gladius_folder=r"...",
    #     out_base=os.path.join(ROOT, "dataset", "newfighter_vs_other"),
    #     rounds=[1, 2, 3],
    #     fighter_names={0: "...", 1: "..."},
    #     fighter_short_names={0: "...", 1: "..."},
    #     identity_marker=None,
    # ),

    # The 4 fights below have no fighter names on file (Gladius tagged them
    # round_id="test" with no name metadata) -- real boxing footage at York
    # Hall, just never matched to a named-export folder like the ones above.
    # Their raw broadcast video + bbox tracking live in Gladius_Data (the
    # pre-rename source export), NOT in the usual drive-download video_folder.
    # tracking_format="raw_bbox" tells prepare_fight_dataset.py to read
    # fighters_tracking_data from tracking_files instead of *_SAM3D_fighterN
    # keypoints_3d (same {frame, bbox} shape either way -- that's all
    # tools/annotate_clips.py actually needs for the overlay/swap check).
    "1st_fight": dict(
        video_folder=(
            r"C:\Users\XRIG\Desktop\Gladius_Data"
            r"\1st_fight-20260510T154005Z-3-001\1st_fight"
        ),
        video_filenames={1: "1.mp4", 2: "2.mp4", 3: "3rd.mp4"},
        gladius_folder=r"C:\Users\XRIG\Desktop\Gladius_Output\1st_fight",
        out_base=os.path.join(ROOT, "dataset", "1st_fight"),
        rounds=[1, 2, 3],
        fighter_names={0: "Fighter 0 (unidentified)", 1: "Fighter 1 (unidentified)"},
        fighter_short_names={0: "Fighter0", 1: "Fighter1"},
        identity_marker=None,
        tracking_format="raw_bbox",
        tracking_folder=(
            r"C:\Users\XRIG\Desktop\Gladius_Data"
            r"\1st_fight-20260510T154005Z-3-001\1st_fight"
        ),
        tracking_files={1: "fight1-round1.json", 2: "fight1-round2.json",
                        3: "fight1-round3.json"},
    ),

    "2nd_fight": dict(
        video_folder=(
            r"C:\Users\XRIG\Desktop\Gladius_Data"
            r"\2nd_fight-20260510T154005Z-3-001\2nd_fight"
        ),
        video_filenames={i: f"{i}.mp4" for i in range(1, 7)},
        gladius_folder=r"C:\Users\XRIG\Desktop\Gladius_Output\2nd_fight",
        out_base=os.path.join(ROOT, "dataset", "2nd_fight"),
        rounds=[1, 2, 3, 4, 5, 6],
        fighter_names={0: "Fighter 0 (unidentified)", 1: "Fighter 1 (unidentified)"},
        fighter_short_names={0: "Fighter0", 1: "Fighter1"},
        identity_marker=None,
        tracking_format="raw_bbox",
        tracking_folder=(
            r"C:\Users\XRIG\Desktop\Gladius_Data"
            r"\2nd_fight-20260510T154005Z-3-001\2nd_fight"
        ),
        tracking_files={i: f"fight2-round{i}.json" for i in range(1, 7)},
    ),

    "3rd_fight": dict(
        video_folder=(
            r"C:\Users\XRIG\Desktop\Gladius_Data"
            r"\3rd_fight-20260510T154007Z-3-001\3rd_fight"
        ),
        video_filenames={i: f"{i}.mp4" for i in range(1, 7)},
        gladius_folder=r"C:\Users\XRIG\Desktop\Gladius_Output\3rd_fight",
        out_base=os.path.join(ROOT, "dataset", "3rd_fight"),
        rounds=[1, 2, 3, 4, 5, 6],
        fighter_names={0: "Fighter 0 (unidentified)", 1: "Fighter 1 (unidentified)"},
        fighter_short_names={0: "Fighter0", 1: "Fighter1"},
        identity_marker=None,
        tracking_format="raw_bbox",
        tracking_folder=(
            r"C:\Users\XRIG\Desktop\Gladius_Data"
            r"\3rd_fight-20260510T154007Z-3-001\3rd_fight"
        ),
        tracking_files={i: f"fight3-round{i}.json" for i in range(1, 7)},
    ),

    "4th_fight": dict(
        video_folder=(
            r"C:\Users\XRIG\Desktop\Gladius_Data"
            r"\4th_fight-20260510T154008Z-3-001\4th_fight"
        ),
        video_filenames={i: f"{i}.mp4" for i in range(1, 7)},
        gladius_folder=r"C:\Users\XRIG\Desktop\Gladius_Output\4th_fight",
        out_base=os.path.join(ROOT, "dataset", "4th_fight"),
        rounds=[1, 2, 3, 4, 5, 6],
        fighter_names={0: "Fighter 0 (unidentified)", 1: "Fighter 1 (unidentified)"},
        fighter_short_names={0: "Fighter0", 1: "Fighter1"},
        identity_marker=None,
        tracking_format="raw_bbox",
        tracking_folder=(
            r"C:\Users\XRIG\Desktop\Gladius_Data"
            r"\4th_fight-20260510T154008Z-3-001\4th_fight"
        ),
        tracking_files={i: f"fight4-round{i}.json" for i in range(1, 7)},
    ),
}


def get_fight(name):
    """Return the config dict for `name`, with derived clips_dir/manifest_path
    filled in (so callers don't each repeat os.path.join(out_base, ...))."""
    if name not in FIGHTS:
        raise KeyError(
            f"Unknown fight {name!r}. Known fights: {list(FIGHTS)}. "
            f"Add a new entry to dataset/fights.py to register one."
        )
    cfg = dict(FIGHTS[name])
    cfg["name"] = name
    cfg.setdefault("clips_dir", os.path.join(cfg["out_base"], "clips"))
    cfg.setdefault("manifest_path", os.path.join(cfg["out_base"], "manifest.json"))
    return cfg


def get_video_path(cfg, round_id):
    """Resolve a round's video file path. Most fights use plain "RoundN.mp4";
    a few (1st_fight/2nd_fight/3rd_fight/4th_fight) have irregular source
    filenames (e.g. "1.mp4", "3rd.mp4") recorded in video_filenames."""
    filenames = cfg.get("video_filenames")
    fname = filenames[round_id] if filenames else f"Round{round_id}.mp4"
    return os.path.join(cfg["video_folder"], fname)


def all_fight_names():
    return list(FIGHTS)
