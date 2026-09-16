#!/bin/bash
# Script: Delete files in "1-audio" that already exist in "run-script-3"
# Use rclone to list files from run-script-3, then delete the duplicate files in 1-audio

REMOTE="ggdrive:"
SRC_FOLDER="1-audio"
REF_FOLDER="run-script-3"

echo "=== Listing files from '${REF_FOLDER}'..."
mapfile -t files < <(rclone lsf "${REMOTE}${REF_FOLDER}/" --files-only)

total=${#files[@]}
echo "=== Found ${total} files in '${REF_FOLDER}'"
echo ""

deleted=0
errors=0

for i in "${!files[@]}"; do
    file="${files[$i]}"
    idx=$((i + 1))
    echo "[${idx}/${total}] Deleting: ${file}"
    
    if rclone deletefile "${REMOTE}${SRC_FOLDER}/${file}" 2>/dev/null; then
        ((deleted++))
    else
        echo "  ⚠ Error or not found: ${file}"
        ((errors++))
    fi
done

echo ""
echo "=== Done ==="
echo "Deleted: ${deleted} files"
echo "Errors/Not found: ${errors} files"
