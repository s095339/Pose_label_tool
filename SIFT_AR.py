import  cv2
import matplotlib.pyplot as plt
import numpy  as np
import natsort
import os




sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


#mycode=================================
path = "./demoVideo/12_bag/"
#img_list = os.listdir(path)
#for i in range(len(img_list)):
#    img_list[i].replace("_Color_", "")


images = os.listdir(path)
images = natsort.natsorted(images)
idx = 0
down_rate = 10

first = False
last_img = None
if __name__ == "__main__" :


    for img_name in images[1:]:

        #mycode read image===================
        if img_name[-4:] != ".png": continue
        if first == False:
            img_pth=os.path.join(path, img_name)

            last_img = cv2.imread(img_pth)
            first = True
            continue


        idx = (idx+1)%down_rate
        if idx != 0: continue
        img_pth=os.path.join(path, img_name)
        print(img_pth)
        frame = cv2.imread(img_pth)



        cam_img = last_img
        ref_img = frame
        print(cam_img.shape)
        print(ref_img.shape)
    # Extract SIFT features from the reference image
        last_img = ref_img


        referenceKeypoints = None
        referenceDescriptors = None

        kpts_1, dscptor_1  = sift.detectAndCompute(ref_img, None)
        kpts_2, dscptor_2  = sift.detectAndCompute(cam_img, None)

        #Match the SIFT features between the reference and camera images

        matches = bf.match(dscptor_1, dscptor_2)
        matches = sorted(matches, key = lambda x:x.distance)

        ref_pts = []
        cam_pts = []


        for match in matches:
        
            ref_pts.append(kpts_1[match.queryIdx].pt)
            cam_pts.append(kpts_2[match.trainIdx].pt)

        ref_pts = np.array(ref_pts)
        cam_pts = np.array(cam_pts)

        #img3 = cv2.drawMatches(ref_img, kpts_1, cam_img, kpts_2, matches[:10], cam_img, matchesThickness = 1)
        #cv2.imshow('SIFT', img3)
        colors = [
            [255,0,0],
            [0,255,0],
            [0,0,255],
            [255,255,0],
            [255,0,255],
            [0,255,255],
            [255,255,255]
        ]
        ref_pts_4 = ref_pts[:7]
        cam_pts_4 = ref_pts[:7]
        ref_img_c = ref_img.copy()
        cam_img_c = cam_img.copy()
        idx = 0
        for pt in ref_pts_4:
            pt = np.array(pt, dtype = np.int32)
            cv2.circle(ref_img_c, pt, 3, colors[idx], 1 )
            idx+=1

        idx = 0
        for pt in cam_pts_4:
            pt = np.array(pt, dtype = np.int32)
            cv2.circle(cam_img_c, pt, 3, colors[idx], 1 )
            idx+=1

        img3 = np.concatenate([cam_img_c, ref_img_c], axis = 1)
        print(img3.shape)
        cv2.imshow("result", img3)
        cv2.waitKey(0)