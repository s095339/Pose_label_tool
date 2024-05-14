#!/usr/bin/bash

echo "bag to avi translator"

echo "enter a bag file name: "
read bag_file


foldername="${bag_file//./_}"
echo $foldername
mkdir $foldername


rs-convert -i $bag_file -p ./${foldername}/

#Actual_Fps = 29978

python png2avi.py --images_path $foldername