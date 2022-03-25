import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

windowName = 'Green screen effect'
trackbarTol = 'Tolerance'
trackbarSoft = 'Softness'
trackbarDefringe = 'Defringe'
start = list()
end = list()

def GreenRemoval(action, x, y, flags, userdata):
    global start,end,startX,endX,startY,endY,minHue,maxHue
    if action == cv2.EVENT_LBUTTONDOWN:
        start = [(x,y)]
    elif action == cv2.EVENT_LBUTTONUP:
        end = [(x,y)]
        startX = min(start[0][0],end[0][0])
        endX = max(start[0][0],end[0][0])
        startY = min(start[0][1],end[0][1])
        endY = max(start[0][1],end[0][1])
        frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        H,S,V = cv2.split(frameHSV[startY:endY,startX:endX])
        minHue = np.min(H)
        maxHue = np.max(H)

def toleranceChange(*args):
    global minHue, maxHue, tolerance
    
    # Get the value from the trackbar 
    tolerance = args[0]
    
    if tolerance != 0:
        # Update minimum and maximum hue values
        if 53 - tolerance*0.5 < 0:
            minHue = 0
        else:
            minHue = 53 - tolerance*0.5
        
        if 56 + tolerance*0.5 > 360:
            maxHue = 360
        else:
            maxHue = 56 + tolerance*0.5


def softnessChange(*args):
    global softness
    
    # Get the value from the trackbar 
    softness = args[0]*10


def defringeChange(*args):
    global defringe
    
    # Get the value from the trackbar
    defringe = args[0]


video = cv2.VideoCapture('greenscreen-demo.mp4')

# Check if camera opened successfully
if (video.isOpened() == False):
    print('Error in reading the video')
    
maximum = 100
tolerance = 0
softness = 0
defringe = 0
startX,endX,startY,endY = 0,0,0,0
minHue,maxHue = 0,360

cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(windowName,GreenRemoval)
cv2.createTrackbar(trackbarTol, windowName, tolerance, maximum, toleranceChange)
cv2.createTrackbar(trackbarSoft, windowName, softness, maximum, softnessChange)
cv2.createTrackbar(trackbarDefringe, windowName, defringe, maximum, defringeChange)

frameNumber = 1
q = 101 # represents character 'e'
background = cv2.imread('background.jpg',cv2.IMREAD_COLOR)
B_back,G_back,R_back = cv2.split(background)

# Read until video is completed
while(video.isOpened()) and (q != 27): # Check if camera opened successfully & identify if 'ESC' is pressed or not
    ret,frame = video.read() # ret checks if video is completed
    copyFrame = frame.copy()
    
    if (ret == True) and (frameNumber == 1):
        cv2.putText(frame,'Select a rectangular patch to remove background and click any key to continue.',
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255), 2)
        cv2.imshow(windowName, frame)       
        q = cv2.waitKey(0) & 0xFF
    
    # Green removal
    threshold = np.array([0,0,0])
    threshold = np.reshape(threshold,[1,1,3])
    
    minVal = np.array([minHue,60,60])
    maxVal = np.array([maxHue,255,255])
    
    frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    truthTable = np.dstack(((frameHSV - minVal) >= threshold, (maxVal - frameHSV) >= threshold))
    B,G,R = cv2.split(frame)
    B = (np.all(truthTable,axis = 2) == False)*1*B
    B1 = (np.all(truthTable,axis = 2) == True)*1*B_back
    G = (np.all(truthTable,axis = 2) == False)*1*G
    G1 = (np.all(truthTable,axis = 2) == True)*1*G_back
    R = (np.all(truthTable,axis = 2) == False)*1*R
    R1 = (np.all(truthTable,axis = 2) == True)*1*R_back
    B += B1
    G += G1
    R += R1
    frame = np.uint8(cv2.merge((B,G,R)))

    # Softness
    frame = cv2.GaussianBlur(frame,(5,5),softness,softness)
    
    # Defringing
    B,G,R = cv2.split(frame)
    G = np.uint8(G*(1 - defringe/100))
    frame = cv2.merge((B,G,R))

    # Increase contrast and output final image
    frame = np.int32(frame)
    frame = np.clip(frame*1.2,0,255)
    frame = np.uint8(frame)
    cv2.imshow(windowName,frame)
    q = cv2.waitKey(5) & 0xFF
    frameNumber += 1
    
video.release()
cv2.destroyAllWindows()