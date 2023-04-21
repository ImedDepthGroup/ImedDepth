#!/bin/bash

# Specify the directories to work with
source_dir="/data/student/data/student/zhouyingquan/ImedDepth/depth/"
mid_dir="/data/student/data/student/zhouyingquan/ImedDepth/data/test/rgb/"
dest_dir="/data/student/data/student/zhouyingquan/ImedDepth/data/test/depth/"

# Loop through each file in the source directory
for filename in $mid_dir/*.png; do
    # Extract just the filename (without the directory path)
    basename=$(basename $filename .png).exr

    # Check if there's a corresponding file in the destination directory
    if [ -e "$source_dir/$basename" ]; then
        # Move the file from the destination directory to the source directory
        mv "$source_dir/$basename" "$dest_dir/"
        echo "Moved $basename from $source_dir to $dest_dir"
    fi
done
