#!/bin/bash
# This script cleans up incomplete experiment runs.
# An experiment is considered incomplete if its directory does not contain
# a 'checkpoints/last.pth' file.

echo "Searching for incomplete experiment directories under the 'runs/' folder..."
echo "--------------------------------------------------------------------"

# Find all potential experiment directories (3 levels deep) and loop through them.
# Using -print0 and read -d '' handles paths with spaces or special characters.
find runs -mindepth 3 -maxdepth 3 -type d -print0 | while IFS= read -r -d '' exp_dir; do
  
  checkpoint_file="${exp_dir}/checkpoints/last.pth"
  
  # Check if the 'last.pth' file does NOT exist.
  if [ ! -f "${checkpoint_file}" ]; then
    echo "Incomplete run found. Deleting directory: ${exp_dir}"
    # The following command will permanently delete the directory and its contents.
    # Please be careful and ensure you have backups if needed.
    rm -rf "${exp_dir}"
  fi
done

echo "--------------------------------------------------------------------"
echo "Cleanup complete."
