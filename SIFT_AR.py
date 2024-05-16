import  cv2
import matplotlib.pyplot as plt
import numpy  as np
try:
    from cv2 import xfeature2d
except:
    pass


sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)



if __name__ == "__main__" :
    cam_img = cv2.imread("./demoVideo/12_bag/_Color_1715768934448.46508789062500.png")
    ref_img = cv2.imread("./demoVideo/12_bag/_Color_1715768939376.69531250000000.png")
    print(cam_img.shape)
    print(ref_img.shape)
# Extract SIFT features from the reference image



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
cameraPose = cv2.findHomography(ref_pts, cam_pts, cv2.RANSAC)


renderedImage = None
renderedImage = cv2.warpPerspective(ref_img, cameraPose[0], cam_img.shape[:2]);
cv2.imshow("Rendered Image", renderedImage);
cv2.imshow("Cam Image", cam_img);
cv2.imshow("ref Image", ref_img);
cv2.waitKey(0);

