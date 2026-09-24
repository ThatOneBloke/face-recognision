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
            

