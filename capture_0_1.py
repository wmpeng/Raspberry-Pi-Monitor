# -*- coding: utf-8 -*-
import cv2
import time
import findDiff
import picamera
import picamera.array
import numpy as np
import io
import datetime
import os
import shutil

lastHour=str(datetime.datetime.now().hour);
lastMonth=str(datetime.datetime.now().month)
lastYear=str(datetime.datetime.now().year)
lastDay=str(datetime.datetime.now().day)

while(True):
    time.sleep(60)
    currHour=str(datetime.datetime.now().hour)
    currMonth=str(datetime.datetime.now().month)
    currYear=str(datetime.datetime.now().year)
    currDay=str(datetime.datetime.now().day)
    if currHour==lastHour:
        continue;
    print("not equal")
    picsPath="./pictures/"+lastYear+"/"+lastMonth+"/"+lastDay+"/"+lastHour
    outPath="./capture/"+lastYear+lastMonth+lastDay+lastHour+".avi";
    lastYear,lastMonth,lastDay,lastHour=currYear,currMonth,currDay,currHour
    
    images = cv2.VideoCapture(picsPath+"/image%d.jpg");
    #fourcc=cv2.VideoWriter_fourcc(*'XVID')
    #out=cv2.VideoWriter("Videotest.avi",'XVID',20.0,(640,480))
    #out = cv2.VideoWriter('out.avi', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'),5.0,(640,480))
    #out = cv2.VideoWriter('out.avi', cv2.cv.CV_FOURCC('X', 'V', 'I', 'D'),20.0,(640,480))
    out = cv2.VideoWriter(outPath,cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'),1.0,(1280,720))

    lastFrame=cv2.imread("pictures/image0.jpg")
    while(images.isOpened()):
        #print("?")
        
        ret, frame = images.read()
        if ret==True:
            # frame = cv2.flip(frame,0)
            # if different then 写入帧
            if(findDiff.isDiff(frame,lastFrame)==False):
                out.write(frame)
            
            lastFrame=frame
            #cv2.imshow('frame',frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        else:
            break

    images.release()
    out.release()
    shutil.rmtree(picsPath);
    
    print("finish one capture")

