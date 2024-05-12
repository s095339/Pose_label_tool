import numpy as np
import cv2
import sys
import time
import pyrealsense2 as rs
import os


from scipy.spatial.transform import Rotation

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

camera_inst = {
      "fx":913.41826508,
      "fy":912.84577345,
      "cx":655.48686401,
      "cy":369.45247047
}

 #p[659.377 371.096]  f[908.057 906.401]
#mycode==================


def draw_box(img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, position_world: np.ndarray, size: np.ndarray, camera_matrix: np.ndarray, distCoeff: np.ndarray):  
    """
    Parameter
    --------------------------------
    img: img to be added bounding box
    tran: transformation from camera coordinate frame to world frame
    position_world: box position with respect to world frame
    size: the size of bounding box
    camera_matrix: instrinsic matrix of camera

    return
    --------------------
    img: img with bounding box
    """
    
    #reference from CenterPose
    width, height, depth = size
    cx, cy, cz = position_world
    # X axis point to the right
    right = cx + width / 2.0
    left = cx - width / 2.0
    # Y axis point upward
    top = cy + height / 2.0
    bottom = cy - height / 2.0
    # Z axis point forward
    front = cz + depth / 2.0
    rear = cz - depth / 2.0
    
    vertices = np.array([
            [cx+0.02, cy, cz+0.02],   # Center
            [left, bottom, rear],  # Rear Bottom Left
            [left, bottom, front],  # Front Bottom Left
            [left, top, rear],  # Rear Top Left
            [left, top, front],  # Front Top Left

            [right, bottom, rear],  # Rear Bottom Right
            [right, bottom, front],  # Front Bottom Right
            [right, top, rear],  # Rear Top Right
            [right, top, front],  # Front Top Right
        ])

    #print(vertices.T.shape)

    

   

    kps = []
    for pt in vertices[:1]:
        pt_proj, _ = cv2.projectPoints(pt, rvec, tvec, camera_matrix, distCoeff)
        #print(pt_proj[0][0])
        pt_proj = pt_proj[0][0].astype(int)
        kps.append(pt_proj)
        cv2.circle(img, pt_proj,2, [0,128,128], 2)
    
    """
    cv2.line(img, kps[1], kps[2], [255,0,0],2)
    cv2.line(img, kps[1], kps[5], [255,0,0],2)
    cv2.line(img, kps[6], kps[2], [255,0,0],2)
    cv2.line(img, kps[5], kps[6], [255,0,0],2)

    cv2.line(img, kps[3], kps[7], [0,255,0],2)
    cv2.line(img, kps[4], kps[8], [0,255,0],2)
    cv2.line(img, kps[3], kps[4], [0,255,0],2)
    cv2.line(img, kps[7], kps[8], [0,255,0],2)

    cv2.line(img, kps[1], kps[3], [0,0,255],2)
    cv2.line(img, kps[4], kps[2], [0,0,255],2)
    cv2.line(img, kps[8], kps[6], [0,0,255],2)
    cv2.line(img, kps[7], kps[5], [0,0,255],2)
    """
    return img
#========================
def aruco_display(corners, ids, rejected, image):
    
	if len(corners) > 0:
		
		ids = ids.flatten()
		
		for (markerCorner, markerID) in zip(corners, ids):
			
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			#print("[Inference] ArUco marker ID: {}".format(markerID))
			

	return image



def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    #print("check0")
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict,
        #parameters=parameters,
        #cameraMatrix=matrix_coefficients,
        #distCoeff=distortion_coefficients
		)
    #print("check1")
        
    
    if len(corners) > 0:
        for i in range(0, len(ids)):
           
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                       distortion_coefficients)
            
            #print(markerPoints)
            cv2.aruco.drawDetectedMarkers(frame, corners) 
            
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  

            #aruco_display(corners, ids, rejected_img_points, frame)

            #mycode======================================================
            #tran = get_transform_matrix(rvec, tvec) # get transform matrix
            draw_box(frame, rvec, tvec, np.array([0.0,0.0,0.0]), np.array([0.015,0.015,0.015]), matrix_coefficients, distortion_coefficients)
            #============================================================

    #print("check2")
    return frame


    

aruco_type = "DICT_5X5_1000"

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters()





#cap = cv2.VideoCapture(0)

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

# Start streaming
profile = pipeline.start(config)
pp = profile.get_stream(rs.stream.color)
intr = pp.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
#distr = pp.as_video_stream_profile().get_distortion()
print(intr)


intrinsic_camera = np.array(
     [
          [camera_inst['fx'],                 0.0,  camera_inst['cx']],
          [0.0,                 camera_inst['fy'],  camera_inst['cx']],
          [0.0,                               0.0,                1.0],
     ], dtype = np.float32
)


try:
    while True:



        

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #print(color_image.shape)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        #depth_colormap_dim = depth_colormap.shape
        #color_colormap_dim = color_image.shape
        output = pose_estimation(color_image, ARUCO_DICT[aruco_type], intrinsic_camera, np.zeros([0,0,0,0,0]))

        cv2.imshow('Estimated Pose', output)


        """
        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        """



        key = cv2.waitKey(1) & 0xFF
    
        if key == ord('q'):
            break

finally:

    # Stop streaming
    pipeline.stop()
    

