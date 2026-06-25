#!/usr/bin/env python3
"""Shared diff/apply logic for merge_returned_manifest.py (single file) and
merge_all_returned.py (batch inbox processor) -- kept in one place so both
tools can never silently drift apart on what counts as a conflict."""


def diff_manifest(live_by_clip, inc_by_clip):
    """Compare one incoming clip-list against the current live state.
    live_by_clip is mutated in place for label/impact_frame fields that are
    NEW (not conflicting) -- callers that want a true dry run should pass a
    deep-enough copy; callers batching multiple incoming files SHOULD let
    this mutate so a second file in the same batch is diffed against the
    first file's additions too (catches teammate-vs-teammate conflicts, not
    just teammate-vs-original-live)."""
    new_labels, frame_added, agree = [], [], []
    conflicts, frame_conflicts, structure_warnings = [], [], []

    for clip_name, inc_c in inc_by_clip.items():
        live_c = live_by_clip.get(clip_name)
        if live_c is None:
            structure_warnings.append(f"'{clip_name}' in incoming but not in live manifest -- skipped")
            continue

        inc_label, live_label = inc_c.get("label"), live_c.get("label")
        if inc_label is not None:
            if live_label is None:
                new_labels.append((clip_name, inc_label))
                live_c["label"] = inc_label  # so a later file in the same batch sees it
            elif live_label == inc_label:
                agree.append(clip_name)
            else:
                conflicts.append((clip_name, live_label, inc_label))

        inc_frame, live_frame = inc_c.get("impact_frame"), live_c.get("impact_frame")
        if inc_frame is not None:
            if live_frame is None:
                frame_added.append((clip_name, inc_frame))
                live_c["impact_frame"] = inc_frame
            elif live_frame != inc_frame:
                frame_conflicts.append((clip_name, live_frame, inc_frame))

    for clip_name in live_by_clip:
        if clip_name not in inc_by_clip:
            structure_warnings.append(f"'{clip_name}' in live but not in incoming -- not touched")

    return {
        "new_labels": new_labels, "agree": agree, "conflicts": conflicts,
        "frame_added": frame_added, "frame_conflicts": frame_conflicts,
        "structure_warnings": structure_warnings,
    }


def print_report(tag, fight, result):
    print(f"\n=== Merge report{tag}: {fight} ===")
    print(f"  New labels (live was unlabeled)  : {len(result['new_labels'])}")
    print(f"  Agreements (same label both)     : {len(result['agree'])}")
    print(f"  CONFLICTS (different label)      : {len(result['conflicts'])}")
    print(f"  New impact_frame marks           : {len(result['frame_added'])}")
    print(f"  CONFLICTING impact_frame marks   : {len(result['frame_conflicts'])}")
    sw = result["structure_warnings"]
    if sw:
        print(f"  Structural warnings              : {len(sw)}")
        for w in sw[:10]:
            print(f"    - {w}")
        if len(sw) > 10:
            print(f"    ... and {len(sw) - 10} more")

    if result["conflicts"]:
        print(f"\n  -- LABEL CONFLICTS (your call, not auto-applied) --")
        for clip_name, live_label, inc_label in result["conflicts"]:
            print(f"    {clip_name}: live={live_label!r}  incoming={inc_label!r}")

    if result["frame_conflicts"]:
        print(f"\n  -- IMPACT_FRAME CONFLICTS (your call, not auto-applied) --")
        for clip_name, live_frame, inc_frame in result["frame_conflicts"]:
            print(f"    {clip_name}: live={live_frame}  incoming={inc_frame}")


def update_manifest_stats(live):
    labeled = sum(1 for c in live["clips"] if c["label"] is not None)
    live["labeled"] = labeled
    live["unlabeled"] = len(live["clips"]) - labeled
    impact_clips = [c for c in live["clips"] if c["label"] == "impact"]
    live["impact_frames_marked"] = sum(1 for c in impact_clips if c.get("impact_frame") is not None)
    live["impact_frames_unmarked"] = sum(1 for c in impact_clips if c.get("impact_frame") is None)
