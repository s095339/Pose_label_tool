import cv2
import numpy as np
import os
import argparse
import sys
import time
from natsort import natsorted
def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str)
    return parser
def png_to_video(img_list, image_path):
    image_folder = images_path
    video_name = f'{image_path}.avi'

    images = img_list
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0,  20, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()




if __name__ == "__main__":
    args = arg().parse_args()
    images_path = args.images_path
    print("get images from ", images_path)

    path = images_path
    #img_list = os.listdir(path)
    #for i in range(len(img_list)):
    #    img_list[i].replace("_Color_", "")


    images = os.listdir(path)
    images = natsorted(images)

    png_to_video(img_list = images, image_path = path)