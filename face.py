import cv2
import sys
import numpy as np


imagePath = 'newpic.png'
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
	gray,
	scaleFactor=1.1,
	minNeighbors=5,
	minSize=(30,30),
	flags=cv2.CASCADE_SCALE_IMAGE
)
for (x,y,w,h) in faces:
	cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 14)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (135,582)
fontScale              = 1
fontColor              = (0,255,0)
lineType               = 2

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
im2 = cv2.resize(image,(590,777))
cv2.putText(im2,'??????', 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)

cv2.imshow("Faces", im2)
cv2.imwrite("facerec.jpg",im2)
cv2.waitKey(0)
