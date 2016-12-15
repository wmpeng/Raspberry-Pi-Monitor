import picamera
import picamera.array
import time
import numpy as np
import cv2
import io
import datetime
import os

def mkdir(path):
    isExists=os.path.exists(path);
    if not isExists:
        os.makedirs(path)
    else:
        return False

def PreviewOpencvJpeg(camera):
    with io.BytesIO() as stream:
        i=0
        lastHour=-1
        #cv2.namedWindow("test2")
        #for frame in camera.capture_continuous(stream, format='jpeg', splitter_port = 2, resize = (640,480), use_video_port=True):
        for frame in camera.capture_continuous(stream, format='jpeg', splitter_port = 1, resize = (1280,720), use_video_port=True):
            #if(i==100):
            #    break;
            currHour=str(datetime.datetime.now().hour)
            currMonth=str(datetime.datetime.now().month)
            currYear=str(datetime.datetime.now().year)
            currDay=str(datetime.datetime.now().day)
            #print str(currMonth)
            path="./pictures/"+currYear+"/"+currMonth+"/"+currDay+"/"+currHour
            if currHour != lastHour :
                mkdir(path)
                lastHour=currHour
                i=0
            
            i=i+1
            data = np.fromstring(frame.getvalue(),dtype=np.uint8)
            #d1 = datetime.datetime.now()
            cv_image = cv2.imdecode(data, 1)
            #d = datetime.datetime.now() - d1
            #print "consuming %dms" % (d.microseconds/1000)
            #print cv_image.shape
            picName=path+"/image"+str(i)+".jpg"
            
            #cv2.putText(cv_image,str(datetime.datetime.now()),(20,20),1,0.4,(0,0,0));
            
            cv2.imwrite(picName,cv_image)
            #imshow("test2",cv_image)
            stream.seek(0)
            stream.truncate(0)
            #cv2.waitKey(10000)
            time.sleep(0.8)
            camera.annotate_text = str(datetime.datetime.now())

with picamera.PiCamera() as camera:
    #camera.resolution = (1920,1080)
    camera.resolution = (1280,720)
    camera.framerate = 25
    #camera.annotate_text = "HKUTANGYU.Inc"
    #print str(datetime.datetime.now())
    camera.vflip = True
    camera.hflip = True
    print "start preview direct from GPU"
    camera.start_preview() # the start_preview() function
    camera.annotate_text = str(datetime.datetime.now())
    PreviewOpencvJpeg(camera)
