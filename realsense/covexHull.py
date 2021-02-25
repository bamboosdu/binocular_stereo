'''
Author: your name
Date: 2021-02-19 10:44:34
LastEditTime: 2021-02-19 13:36:03
LastEditors: Please set LastEditors
Description: In User Settings Edit

FilePath: /binocular_stereo/python/covexHull.py
'''
import cv2
import numpy as np

img=np.zeros((512,512,3),np.uint8)

pts=np.array([[200,250],[250,300],[300,270],[270,200],[120,240]],np.int32)

pts=pts.reshape((-1,1,2))

cv2.fillPoly(img,[pts],(255,255,255))

cv2.imshow("hello",img)
cv2.waitKey(0)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

contours,hierarchy=cv2.findContours(thresh,2,1)


cnt=contours[0]
print("Lengh of contours is ",len(contours))

hull=cv2.convexHull(cnt)

length=len(hull)
for i in range(len(hull)):
    cv2.line(img,tuple(hull[i][0]),tuple(hull[(i+1)%length][0]),(0,255,0),2)

cv2.imshow('line',img)
cv2.waitKey()