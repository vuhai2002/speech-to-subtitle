#!/usr/bin/env python3
"""
Copy the 300 shortest (by duration) audio files from GDrive 1-audio to run-script-3.
Uses rclone mount + ffprobe to read duration efficiently.
"""

import subprocess
import json
import os
import sys
import time
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed

REMOTE = "ggdrive:"
SRC_FOLDER = "1-audio"
DST_FOLDER = "run-script-3"
MOUNT_POINT = "/tmp/gdrive_mount"
TOP_N = 300
WORKERS = 5  # parallel workers for getting duration

def mount_gdrive():
    """Mount GDrive using rclone."""
    os.makedirs(MOUNT_POINT, exist_ok=True)
    print(f"🔗 Mounting {REMOTE}{SRC_FOLDER}/ to {MOUNT_POINT}...")
    
    proc = subprocess.Popen(
        ["rclone", "mount", f"{REMOTE}{SRC_FOLDER}/", MOUNT_POINT,
         "--read-only",
         "--vfs-cache-mode", "off",
         "--no-checksum",
         "--dir-cache-time", "5m",
         "--daemon"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    proc.wait()
    
    # Wait for mount to be ready
    for i in range(30):
        if os.path.ismount(MOUNT_POINT) or os.listdir(MOUNT_POINT):
            print(f"   ✅ Mounted successfully!")
            return True
        time.sleep(1)
    
    print("   ❌ Mount timeout!")
    return False

def unmount_gdrive():
    """Unmount GDrive."""
    print("\n🔗 Unmounting...")
    subprocess.run(["fusermount", "-uz", MOUNT_POINT], capture_output=True)
    print("   ✅ Unmounted")

def get_duration(filepath):
    """Get duration of an audio file using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe",
             "-v", "quiet",
             "-print_format", "json",
             "-show_format",
             filepath],
            capture_output=True, text=True, timeout=120
        )
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        return os.path.basename(filepath), duration
    except Exception as e:
        return os.path.basename(filepath), float('inf')

def format_duration(seconds):
    """Format seconds to HH:MM:SS."""
    if seconds == float('inf'):
        return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

def main():
    # Step 1: Mount GDrive
    if not mount_gdrive():
        sys.exit(1)

    try:
        # Step 2: Get file list
        files = [f for f in os.listdir(MOUNT_POINT)
                 if os.path.isfile(os.path.join(MOUNT_POINT, f))]
        print(f"📋 Found {len(files)} files\n")

        # Step 3: Get duration for each file (parallel)
        print(f"⏱️  Getting duration for {len(files)} files ({WORKERS} workers)...")
        durations = []
        completed = 0
        errors = 0

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {}
            for f in files:
                fp = os.path.join(MOUNT_POINT, f)
                futures[executor.submit(get_duration, fp)] = f

            for future in as_completed(futures):
                filename, duration = future.result()
                durations.append((filename, duration))
                completed += 1
                if duration == float('inf'):
                    errors += 1
                if completed % 50 == 0 or completed == len(files):
                    print(f"   Progress: {completed}/{len(files)} (errors: {errors})")

        # Step 4: Sort by duration and take shortest TOP_N
        durations.sort(key=lambda x: x[1])
        # Remove files with inf duration
        valid = [(f, d) for f, d in durations if d != float('inf')]
        print(f"\n   Valid files with duration: {len(valid)}")
        
        shortest = valid[:TOP_N]

        print(f"\n📊 Top {TOP_N} shortest files:")
        print(f"   Shortest: {shortest[0][0]} ({format_duration(shortest[0][1])})")
        print(f"   Longest of top {TOP_N}: {shortest[-1][0]} ({format_duration(shortest[-1][1])})")

        # Save list for reference
        with open("shortest_300.txt", "w") as f:
            for filename, duration in shortest:
                f.write(f"{format_duration(duration)}\t{filename}\n")
        print(f"   📝 Saved list to shortest_300.txt")

    finally:
        unmount_gdrive()

    # Step 5: Copy files to destination using rclone server-side copy
    print(f"\n📤 Copying {len(shortest)} files to {REMOTE}{DST_FOLDER}/...")
    
    for i, (filename, duration) in enumerate(shortest, 1):
        if i % 20 == 1 or i == len(shortest):
            print(f"   [{i}/{len(shortest)}] {filename} ({format_duration(duration)})")
        
        result = subprocess.run(
            ["rclone", "copyto",
             f"{REMOTE}{SRC_FOLDER}/{filename}",
             f"{REMOTE}{DST_FOLDER}/{filename}"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"   ❌ Error: {filename}: {result.stderr.strip()}")

    print(f"\n✅ Done! Copied {len(shortest)} shortest files to {REMOTE}{DST_FOLDER}/")

if __name__ == "__main__":
    main()
