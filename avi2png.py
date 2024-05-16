import cv2
import os

vid_name = '12_bag.avi'
vidcap = cv2.VideoCapture(vid_name)
success,image = vidcap.read()
count = 0
if not os.path.exists(vid_name): os.mkdir(vid_name[:-4])
print(f"frame_{count:04}")
while success:
  cv2.imwrite(os.path.join( vid_name[:-4]  ,f"frame_{count:04}.png"), image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1