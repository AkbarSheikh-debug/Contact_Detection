"""
Mirror Images Script
====================
Horizontally flips all JPEG images from the input directory tree,
saving mirrored versions into a parallel output directory,
preserving the exact same folder structure.

Input  : multi_test_1
Output : multi_test_1_mirror
"""

import os
from PIL import Image

# ==============================
# CONFIG
# ==============================
INPUT_DIR  = r"C:\Users\XRIG\Desktop\New_Blender_Sprints\Synthetic_dataset\multi_test_2"
OUTPUT_DIR = r"C:\Users\XRIG\Desktop\New_Blender_Sprints\Synthetic_dataset\multi_test_2_mirror"

# File extensions to process
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ==============================

def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        return

    processed = 0
    skipped = 0
    errors = 0

    for root, dirs, files in os.walk(INPUT_DIR):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            src = os.path.join(root, fname)

            # Recreate the same relative path inside OUTPUT_DIR
            rel = os.path.relpath(src, INPUT_DIR)
            dst = os.path.join(OUTPUT_DIR, rel)

            if os.path.exists(dst):
                skipped += 1
                continue

            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with Image.open(src) as img:
                    mirrored = img.transpose(Image.FLIP_LEFT_RIGHT)
                    if dst.lower().endswith((".jpg", ".jpeg")):
                        mirrored.save(dst, "JPEG", quality=65)
                    else:
                        mirrored.save(dst)

                processed += 1
                if processed % 500 == 0:
                    print(f"  [OK] {processed} images mirrored so far...")

            except Exception as e:
                print(f"  [ERR] Error processing {src}: {e}")
                errors += 1

    print(f"\nDone.")
    print(f"   Input    : {INPUT_DIR}")
    print(f"   Output   : {OUTPUT_DIR}")
    print(f"   Mirrored : {processed}")
    print(f"   Skipped  : {skipped}  (already existed)")
    print(f"   Errors   : {errors}")

if __name__ == "__main__":
    main()