#!/usr/bin/ python
# -*- coding: utf-8 -*- 

# 使用視覺方法測量目標在世界坐標系中的坐標
# 首先估計相機姿態,然後測算目標marker中心點在世界坐標系中的位置.
# 使用方法:
# 1. 相機校準,
# 2. 在空間中放置4個以上的基準坐標點,在程序中給定這些點的信息,包括ID和世界坐標
# 3. 被測目標使用marker標記,在程序中給定這些點的markerID
# 4. 拍攝錄像,確保4個標志點在視野內.
# 5. 運行程序處理視頻幀
# CR@ Guofeng, mailto:gf@gfshen.cn
# 
# ------版本歷史---
# ---V1.0
# ---2019年7月19日
#    初次編寫



import numpy as np
import cv2
import cv2.aruco as aruco

#mycode
import os
import glob
import natsort




def estimateCameraPose(cameraMtx, dist, refMarkerArray,corners,markerIDs):
    '''
    根據基準點的marker，解算相機的旋轉向量rvecs和平移向量tvecs，(solvePnP(）實現)
    並將rvecs轉換爲旋轉矩陣輸出(通過Rodrigues())
    輸入：
        cameraMtx內參矩陣，
        dist畸變係數。
        當前處理的圖像幀frame，
        用於定位世界座標系的參考點refMarkerArray.  py字典類型,需要len(refMarkerArray)>=3, 格式：{ID:[X, Y, Z], ID:[X,Y,Z]..}
        corners, detectMarkers()函數的輸出
        markerIDs, detectMarkers()函數的輸出
    輸出：旋轉矩陣rMatrix, 平移向量tVecs
    '''
    marker_count = len(refMarkerArray)
    #if marker_count<4: #標誌板少於四個
    #    raise RuntimeError('at least 3 pair of points required when invoking solvePnP')
    

    corners=corners; ids=markerIDs
    print('ids:\n')
    print(ids)
    print('corners:\n')
    print(corners)

    objectPoints=[]
    imagePoints=[]
    #檢查是否探測到了所有預期的基準marker
    if len(ids) !=0: #檢測到了marker,存儲marker的世界坐標到objectPoints，構建對應的圖像平面坐標列表 imagePoints
        print('------detected ref markers----')
        for i in range(len(ids)): #遍歷探測到的marker ID,
            if ids[i][0] in refMarkerArray: #如果是參考點的標志，提取基準點的圖像座標，用於構建solvePnP()的輸入

                print('id:\n ' + str(ids[i][0]))
                print('cornors: \n '+ str(corners[i][0]))
                objectPoints.append(refMarkerArray[ ids[i][0] ])
                imagePoints.append(corners[i][0][0].tolist()) #提取marker的左上點
        objectPoints=np.float32(objectPoints)
        imagePoints=np.float32(imagePoints)
            
        print('------------------------------\n')
        print('objectPoints:\n'+str(objectPoints))
        print('imagePoints:\n'+str(imagePoints))
        pass
    else:
        return False, None, None

    #如果檢測到的基準參考點大於3個，可以解算相機的姿態啦
    if len(objectPoints)>=4:
        #至少需要4個點
        retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMtx, dist)
        #rMatrix,jacobian = cv2.Rodrigues(rvec)
        return True, rvec, tvec
    else:
        return False, None, None


    #返回值
    #return rMatrix=[], tVecs=[]



def detectTarget(cameraMatrix, dist, rMatrix, tvec, targetMarker, corners, markerIDs,zWorld = 0.0):
    '''
    測算目標marker中心在世界坐標系中的位置
    輸入:

    輸出:
        與markerIDs長度相等的列表,包含位置確定的目標坐標,未檢測到填None,例如[None,[x2,y2,z2]]
    '''
    if rMatrix==[]:
        return
    targets_count=len(targetMarker)
    if targets_count == 0:
        raise Exception('targets empty, areyou dou?')

    #創建與targetMarker相同尺寸的列表,用於存儲解算所得到目標的世界坐標
    targetsWorldPoint=[None] * targets_count

    for i in range(len(markerIDs)): #遍歷探測到的marker ID,
        markerIDThisIterate = markerIDs[i][0]
        if markerIDThisIterate in targetMarker: #如果是目標marker的ID
            #獲得當前處理的marker在targetMarker中的下標,用於填充targetsWorldPoint
            targetIndex = targetMarker.index(markerIDThisIterate)
        else:
            continue
        #計算marker中心的圖像坐標
        markerCenter = corners[i][0].sum(0)/4.0
        #畸變較正,轉換到相機坐標系,得到(u,v,1)
        #https://stackoverflow.com/questions/39394785/opencv-get-3d-coordinates-from-2d
        markerCenterIdeal=cv2.undistortPoints(markerCenter.reshape([1,-1,2]),cameraMatrix,dist)
        markerCameraCoodinate=np.append(markerCenterIdeal[0][0],[1])
        print('++++++++markerCameraCoodinate')
        print(markerCameraCoodinate)

        #marker的坐標從相機轉換到世界坐標
        markerWorldCoodinate = np.linalg.inv(rMatrix).dot((markerCameraCoodinate-tvec.reshape(3)) )
        print('++++++++markerworldCoodinate')
        print(markerWorldCoodinate)
        #將相機的坐標原點轉換到世界坐標系
        originWorldCoodinate = np.linalg.inv(rMatrix).dot((np.array([0, 0, 0.0])-tvec.reshape(3)) )
        #兩點確定了一條直線 (x-x0)/(x0-x1) = (y-y0)/(y0-y1) = (z-z0)/(z0-z1) 
        #當z=0時,算得x,y
        delta = originWorldCoodinate-markerWorldCoodinate
        #zWorld = 0.0
        xWorld = (zWorld-originWorldCoodinate[2])/delta[2] * delta[0] + originWorldCoodinate[0]
        yWorld = (zWorld-originWorldCoodinate[2])/delta[2] * delta[1] + originWorldCoodinate[1]
        targetsWorldPoint[targetIndex]=[xWorld,yWorld,zWorld]
        
        print('-=-=-=\n Target Position '+ str(targetsWorldPoint[targetIndex]) )
        pass
    return targetsWorldPoint



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
            [cx, cy, cz],   # Center
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
    for pt in vertices:
        pt_proj, _ = cv2.projectPoints(pt, rvec, tvec, camera_matrix, distCoeff)
        #print(pt_proj[0][0])
        pt_proj = pt_proj[0][0].astype(int)
        kps.append(pt_proj)
        cv2.circle(img, pt_proj,2, [0,128,128], 2)
    

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

    return img


camera_inst = {
      "fx":642.905,
      "fy":642.905,
      "cx":648.156,
      "cy":359.988
}


if __name__ == '__main__':
    
    mtx = np.array(
        [
            [camera_inst['fx'],                 0.0,  camera_inst['cx']],
            [0.0,                 camera_inst['fy'],  camera_inst['cy']],
            [0.0,                               0.0,                1.0],
        ], dtype = np.float32
    )
    dist = np.zeros([0,0,0,0,0], dtype = np.float32)#npzfile['dist']


    #保存基準點的信息,檢測到之後會更新.
    rMatrix=[]
    tvec=[]
    #######

    #處理視頻畫面 
   

    ##process and measure target position
    #0.1. 指定基準點的marker ID和世界坐標
    # [[marker ID, X, Y, Z]..]
    refMarkerArray={   \
        1: [0.5, 0.0, 0.0], \
        2: [0.0, 0.0, 0.0], \
        3: [0.0, 0.0, 0.7], \
        4: [0.5, 0.0, 0.7], \
        5: [0.25,0.0, 0.7],\
        6: [0.25,0.0, 0.0],\
        9: [0.0, 0.0, 0.35],\
        8: [0.55,0.5,0.0],\
        7: [0.55,0.5, 0.7],
        10:[0.55, 0.35, 0],
        12:[0.55, 0.35, 0.7]

    }
    #0.2 指定目標的markr ID
    #targetMarker =[10,11]

    #mycode=================================
    path = "./demoVideo/12_bag/"
    #img_list = os.listdir(path)
    #for i in range(len(img_list)):
    #    img_list[i].replace("_Color_", "")


    images = os.listdir(path)
    images = natsort.natsorted(images)
    idx = 0
    down_rate = 10
    #=======================================
    #mycode
    for img_name in images:

        #mycode read image===================
        if img_name[-4:] != ".png": continue

        idx = (idx+1)%down_rate
        if idx != 0: continue
        img_pth=os.path.join(path, img_name)
        print(img_pth)
        frame = cv2.imread(img_pth)

        #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('image', 1280,720)
        #cv2.imshow("image",frame)
        #====================================
        #1. 估計camera pose 
        #1.1 detect aruco markers
        img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        parameters = aruco.DetectorParameters_create()
    
        
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=parameters)
        aruco.drawDetectedMarkers(frame, corners) #Draw A square around the markers
        
        #1.2 estimate camera posec
        try:
            gotCameraPose, rvec, tvec = estimateCameraPose(mtx, dist, refMarkerArray,corners,ids)
        except:
            continue
        #1.3 updata R, T to static value 
        if gotCameraPose: 
            print('rvec\n'+str(rvec))
            print('tvec\n'+str(tvec))

        #2. 根據目標的marker來計算世界坐標系坐標
        #detectTarget(mtx, dist, rMatrix, tvec, targetMarker, corners, ids)
        
            frame = draw_box(frame, rvec, tvec, np.array([0.4,0.1,0.5]), np.array([0.2,0.2,0.2]), mtx, dist)

            cv2.namedWindow('detect',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('detect', 1280,720)
            cv2.imshow("detect",frame)
            cv2.waitKey(0)
        
        #cap.release()
        cv2.destroyAllWindows()