import cv2
import sys
import numpy as np
import os

mlmodel = "Face_Recognision/haarcascade_frontalface_default.xml"
datasets = "Face_Recognision/datasets"
print("recognising your face, please ensure suficient lighting...")

(images, labels, names, id) = ([], [], {}, 0)
#at the time of recognising, multiple images will be stored in the images list, and the names of the people in the labels list. then in the dictionary will hold the images and labels together. the ID is whether it is able to capture the image or not.

for (subdirs, dirs, files) in os.walk(datasets):
    #subdirs is the folder of dataset, dirs is the specific folders (xavier, michal, mom) and files are the photo
    for subdir in dirs:
        names[id] = subdir
        #the names of the file with the [0] (meaning it will use the first image) will go into the subdir variable
        subjectpath = os.path.join(datasets, subdir)
        #this will then create the complete path of files
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            #this will join every path with the file name
            label = id
            img = cv2.imread(path, 0)
            images.append(img)
            labels.append(label)
            #adds img and label to images and labels lists
        id += 1
        #increasing the id number which serves as the index for the two lists of images labels
    images = np.array(images)
    labels = np.array(labels)
    #this converts the list into a numberical python library, holding it in an array (a list of numbers). this we need to use numpy to the upcoming code, and can only do this now and not later
    
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, labels)
#this creates the machine learning model, and loads it in a variable. then we give it the images to recognise trhe faces. the machine only learns in the numpy variant. that is why we changed it#
faces = cv2.CascadeClassifier(mlmodel)
webcam = cv2.VideoCapture(0)

while True:
    (_, im) = webcam.read()
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    myface = faces.detectMultiScale(grey, 1.3, 5)

    for (x, y, w, h) in myface:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 10)
        #this will create a rectangle around the captured face. (x, y) are the coords, (x + w, y + h) is the width and height since x and y are the same as width and height. (0, 0, 255) is the colour red in BGR and 10 is the thickness of the line
        face = grey[y:y + h, x:x + w]
        #ensures proper face size for recognision live
        face_resize = cv2.resize(face, (130, 100))
        prediction = recognizer.predict(face_resize)
        #it will predict who is on the camera from what it has been given. Recogniser will give two numbers. we only need the second one, so that is why you see prediction[1] in the next line
        if prediction[1] < 500:
            #less than 500 is a good prediction, and is able to predict. 
            confidence = int(100 * (1 - (prediction[1] / 300))) 
            #this will say how confident it is that it is correct
            cv2.putText(im, f'{names[prediction[0]]} - {confidence}%', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            #this will put at text on the image to show how confident who it is (precentage)
        else:
            cv2.putText(im, 'Not recognized', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    cv2.imshow("Face Recognision", im)
    key = cv2.waitKey(10)
    if key == 27:
        #27 is the espcape key
        break

webcam.release()

cv2.destroyAllWindows()
#it will destroy all windows