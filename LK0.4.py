# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:14:36 2021

@author: 15669
"""

import cv2 as cv
import numpy as np
def eli_white (img):
    h,w,c = img.shape
    for row in range(h):
        for col in range(w):
            if img[row,col,0] >= 80 and img[row,col,1] >= 80 and img[row,col,2] >= 80:
                img[row,col] = (0,0,0)
    return img
#消除白光，img.shape得到图像基本信息，三个通道，前两个是row和col第三个是颜色通道，RGB对应2，1，0（倒过来），现在是以80为阈值。 
#谨慎使用此算法，访问每个像素点会使程序运行十分慢


def HSV(img):
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    #
    lowred1 = np.array([0,43,46])
    upred1 = np.array([10,255,255])
    lowred2 = np.array([156,43,46])
    upred2 = np.array([180,255,255])
    #红色的HSV值范围
    mask1 = cv.inRange(hsv,lowred1,upred1)
    mask2 = cv.inRange(hsv,lowred2,upred2)
    return mask1+mask2



def binary (img):
    img[:,:,0] = 0
    img[:,:,1] = 0
    ret,img = cv.threshold(img,100,255,cv.THRESH_BINARY)

    grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    ret,img_r = cv.threshold(grey,50,255,cv.THRESH_BINARY)
    return img_r
#二值化


def expend(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.dilate(img, kernel,3)
    return dst
#膨胀函数，为了消除离散值，3是次数



#def isspeed():   #输入一个list,list中包含矩阵
    

def rank(contours,hierarchy):
    m = []
    p = {}
    n = 0
    j = 0
    flag = False
    #print(hierarchy)
    for i in hierarchy[0]:        
        #print(n)
        #print(p)
        if i[3] != -1:            
            if i[3] in p:
                #print(p)
                p[i[3]][0] = p[i[3]][0] + 1
            else:
                p[i[3]] = [1,n]
                #print(p[i[3]])
        n=n+1
    if len(p) != 0:
        for i in p:
            if type(p[i]) != list:
                continue
            if p[i][0] == 1:
                flag = True
                m.append(p[i][1])
    if flag:
        for i in m:
            if cv.contourArea(contours[i]) >= cv.contourArea(contours[m[0]]):   #cv.contourArea用来确定轮廓面积
                j = i
            return j
            break
    else:
        return findthetarget(contours,hierarchy)
#利用子轮廓和父轮廓作为检测限制条件，通过观察我们要找到的轮廓是只有一个夫轮廓的，找出所以符合条件的轮廓，然后通过比较面积大小来确定最终轮廓


def findthetarget(contours,hierarchy):
    key= []
    t=0
    for i in contours:
        a = cv.approxPolyDP(i,5, True)  #cv.approxPolyDP是用来确定所求轮廓的形状大小，len(a)则为包括圈数

        if (len(a) >= 4 and len(a) <= 5):
            key.append(t)
        t = t+1
    n=0
    for i in key:
        if cv.contourArea(contours[i]) >= cv.contourArea(contours[key[0]]):
            n = i
    return n             
    
def center(img):
    c = cv.moments(img)
    cx = int(c["m10"]/c["m00"])
    cy = int(c["m01"]/c["m00"])
    return cx,cy


cap = cv.VideoCapture("D://test//test3.mp4")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


out = cv.VideoWriter('out.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))  # 保存视频



while cap.isOpened():
    ret,frame = cap.read()
    if ret == True:
    
        img = frame.copy()
        #img = eli_white(img)
        img = HSV(img)
        img = expend(img)
        contours,hierarchy = cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    #if type(hierarchy) ==  np.ndarray:
        n = rank(contours,hierarchy)
        cv.drawContours(frame,contours,n,(0,255,0),3)
    
    #if ret == True:


        #out.write(frame)  #视频写入
    #:
        #break
        x,y = center(contours[n])
        cv.circle(frame,(x,y),3,(255,255,255),-1)
        cv.putText(frame,"center",(x-20,y-20),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv.imshow("frame",frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
       break




cap.release()
#out.release()
cv.destroyAllWindows()





