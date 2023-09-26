import cv2
import math
import os

count = 0
videoFile = "MMBCReco/dataManip/videos/bubbleChamber.mp4"
#framePath = "MMBCReco/dataManip/frames"
framePath = "MMBCReco/src/GAN/trainGAN"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5)

while((cap.isOpened())):
    frameId = cap.get(1) #frame num
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename = framePath + "/chamber1frame%d.jpg" % count
        #for cropping
        #frame = frame[1080:2000, 0:2000]
        cv2.imwrite(filename, frame)
        print("Frame: %d saved" % count)
        count+=1

cap.release()
print("Done")