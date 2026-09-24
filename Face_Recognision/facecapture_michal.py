import cv2
import sys
import os
import time

mlmodel = "Face_Recognision/haarcascade_frontalface_default.xml"
dataset = "Face_Recognision/datasets"
subfolder = "michal" 
path = os.path.join(dataset, subfolder)
#joins the path of data set and sub folder, which will take the images which are captured and saved to the "Xavier" folder which is inside dataset.
width = 130
height = 100
facecapture = cv2.CascadeClassifier(mlmodel)
#this will apply the machine learning model to a data set using cascade function
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("cannot open webcam.")
    sys.exit()
#this will ensure the webcam only captures the face.
count = 1
#this variable will count the number of images taken by the camera.

while count < 30:
    (_, im) = webcam.read()
    #the _ is because the function takes two variables. the red function will give the name of the image (which is why we have an _), and the image, which is what the variable "im" represents.
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #the computer will only accept a black and white image, since 1 is black, and 0 is white. (Binary)
    faces = facecapture.detectMultiScale(grey, 1.3, 4)
    #this will assign different values to the grey colour (shades)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 10)
        #this will create a rectangle around the captured face. (x, y) are the coords, (x + w, y + h) is the width and height since x and y are the same as width and height. (0, 0, 255) is the colour red in BGR and 10 is the thickness of the lines
        face = grey[y:y+h, x:x+w]
        #it will measure the x and y co-ordinates and the width and height in grey scale
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('% s/% s.png'%(path,count),face_resize)
        #this will give the name to the file on its own, so we do not have to write it down for it.
        count += 1
    cv2.imshow("opencv", im)
    cv2.waitKey(10)

webcam.release()
cv2.destroyAllWindows()