#!/bin/bash

input_dir="/home/khw/IBVS-interbotix/img"
output_dir="./assets/vid"

mkdir -p "$output_dir"

ffmpeg -i $input_dir/rgbimage_%d.png -c:v libx264 -strict -2 -preset slow -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -f mp4 $output_dir/output1.mp4
