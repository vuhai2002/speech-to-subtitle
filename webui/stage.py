"""Copy a single audio file into a staging directory (used to feed the dir-batch Vertex script)."""
import shutil
import sys
from pathlib import Path

if __name__ == "__main__":
    src, dst_dir = sys.argv[1], sys.argv[2]
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, Path(dst_dir) / Path(src).name)
    print(f"[stage] {src} -> {dst_dir}")
