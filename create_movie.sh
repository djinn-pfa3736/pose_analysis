#! /bin/bash

ffmpeg -r 30 -i compare_%04d.png -vcodec libx264 -pix_fmt yuv420p -r 60 $1
rm *.png