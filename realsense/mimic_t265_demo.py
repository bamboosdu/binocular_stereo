'''
Author: your name
Date: 2021-01-04 14:57:25
LastEditTime: 2021-01-04 19:12:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PSMNet/home/zq/zq/github/binocular_stereo/realsense/mimic_t265_demo.py
'''


import pyrealsense2 as rs

import cv2
import numpy as np
from math import tan,pi


pipe=rs.pipeline()

cfg=rs.config()
cfg.enable_stream(rs.stream.pose)

pipe.start(cfg)

try:
    WINDOW_TITLE='Realsense'
    cv2.nameWindow(WINDOW_TITLE,cv2.WINDOW_NORMAL)


    window_size=5
    min_disp=0

    num_disp=112-min_disp
    max_disp=min_disp+num_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                                   numDisparities = num_disp,
                                   blockSize = 16,
                                   P1 = 8*3*window_size**2,
                                   P2 = 32*3*window_size**2,
                                   disp12MaxDiff = 1,
                                   uniquenessRatio = 10,
                                   speckleWindowSize = 100,
                                   speckleRange = 50)

    
    profiles=pipe.get_active_profile()
    
    streams={"left": profiles.get_stream(rs.stream.fisheye,1).as_video_stream_profile(),
              "right": profiles.get_stream(rs.stream.fisheye,2).as_video_stream_profile()}
    
    intrinsics={"left":stream["left"].get_intrinsics(),
                "right": stream["right"].get_intrinsics()}
    
    print("Left camera:",intrinsics["left"])
    print("Right camera:", intrinsics["right"])
    """
    Left camera: [ 848x800  p[422.386 399.522]  f[285.518 285.516]  Kannala Brandt4 [-0.00338519 0.0419216 -0.0399955 0.00711742 0] ]
    Right camera: [ 848x800  p[419.602 403.394]  f[285.499 285.648]  Kannala Brandt4 [0.00114178 0.0288536 -0.026192 0.00248861 0] ]
    
    """

    K_left  = camera_matrix(intrinsics["left"])
    D_left  = fisheye_distortion(intrinsics["left"])
    K_right = camera_matrix(intrinsics["right"])
    D_right = fisheye_distortion(intrinsics["right"])
    (width, height) = (intrinsics["left"].width, intrinsics["left"].height)

    # Get the relative extrinsics between the left and right camera
    (R, T) = get_extrinsics(streams["left"], streams["right"])

    stereo_fov_rad=90*(pi/180)
    stereo_height_px=300
    stereo_focal_px=stereo_height_px/2 /tan(stereo_fov_rad/2)

    #The rotation between the cameras
    R_left=np.eye(3)
    R_right=R
    

    stereo_width_px=stereo_height_px+max_disp
    stereo_size=(stereo_width_px,stereo_height_px)
    stereo_cx=(stereo_height_px-1)/2+max_disp
    stereo_cy=(stereo_height_py-1)/2

    #Construct the left and right projection matrices, the only difference 
    P_left=np.array([[stereo_focal_px,0,stereo_cx,0],
                     [0,stereo_focal_px,stereo_cy,0],
                     [0,0,1,0]])
    P_right=P_left.copy()

    P_right[0][3]=T[0]*stereo_focal_px

    #Construct Q for use with cv2.reprojectImageTo3D.
    #Substract max_disp from x since we will crop the disparity later

    
    #create an undistortion map for the left and right camera which applies the rectification and undos the camera distortion
    mltype=cv2.CV_32FC1
    (lm1,lm2)=cv2.fisheye.initUndistortRectifyMap(K_left,D_left,R_left,P_left,stereo_size,mltype)
    (rml,rm2)=cv2.fisheye.initUndistortRectifyMap(K_right,D_right,R_right,P_right,stereo_size,mltype)

    undistort_rectify={"left":(lm1,lm2),
                        "right":(rm1,rm2)}
    
    mode="stack"
    while True:
        frame_mutex.acquire()
        valid=frame_data["timestamp_ms"] is not None
        frame_mutex.release()

        if valid:
            frame_mutex.acquire()
            frame_copy={"left" : frame_data["left"].copy(),
                         "right" : frame_data["right"].copy()}
            frame_mutex.release()

            # Undistort and crop the center of the frames
            center_undistorted = {"left" : cv2.remap(src = frame_copy["left"],
                                          map1 = undistort_rectify["left"][0],
                                          map2 = undistort_rectify["left"][1],
                                          interpolation = cv2.INTER_LINEAR),
                                  "right" : cv2.remap(src = frame_copy["right"],
                                          map1 = undistort_rectify["right"][0],
                                          map2 = undistort_rectify["right"][1],
                                          interpolation = cv2.INTER_LINEAR)}
            
            #compute the disparity on the center of the frames and convert it to a pixel disparity
            disparity =stereo.compute(center_undistorted["left"],center_undistorted["right"]).astype(np.float32)/16.0
            
            #re-crop juet the valid part of the disparity
            disparity=disparity[:,max_disp:]
            
            #convert disparity to 0-255 and color it
            disp_vis=255*(disparity-min_disp)/num_disp
            disp_color=cv2.applyColorMap(cv2.convertScaleAbs(disp_vis,1),cv2.COLOR_GRAY2RGB)


    


    