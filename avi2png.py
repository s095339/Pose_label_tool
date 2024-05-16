import cv2
import os

vid_name = '12_bag.avi'
vidcap = cv2.VideoCapture(vid_name)
success,image = vidcap.read()
count = 0
os.mkdir(vid_name[:-4])

while success:
  cv2.imwrite(os.path.join( vid_name[:-4]  ,"%d.png" % count), image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1