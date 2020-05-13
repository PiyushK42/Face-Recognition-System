##Author:Amartya Kalapahar
##Project: Absolute Face Technologies Internship Assignment

# We will import openCV library for image processing, opening the webcam etc
#Os is required for managing files like directories
#Numpy is basically used for matrix operations
#PIL is Python Image Library

import cv2
import os
import re
import numpy as np
from PIL import Image


#Method for checking existence of path i.e the directory
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# We will be using Local Binary Patterns Histograms for face recognization since it's quite accurate than the rest
recognizer = cv2.face.LBPHFaceRecognizer_create()

# For detecting the faces in each frame we will use Haarcascade Frontal Face default classifier of OpenCV
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

#method getting the images and label data

def getImagesAndLabels(path):

    # Getting all file paths
    #imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    #empty face sample initialised
    faceSamples=[]
    
    # IDS for each individual
    ids = []
    names=[]

    # Looping through all the file path
    #for imagePath in imagePaths:
    for subdir, dirs, files in os.walk(path):
        for file in files:
            imagePath=os.path.join(subdir,file)

            # converting image to grayscale
            PIL_img = Image.open(imagePath).convert('L')

            # converting PIL image to numpy array using array() method of numpy
            img_numpy = np.array(PIL_img,'uint8')

            # Getting the image id
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            #id=int(re.search('.(.*).',imagePath).group(1))
            #name=re.search('/(.*).',imagePath).group(1)
            
            #name = str(os.path.split(imagePath)[-2].split(".")[0])

             # Getting the face from the training images
            faces = detector.detectMultiScale(img_numpy)

            # Looping for each face and appending it to their respective IDs
            for (x,y,w,h) in faces:

                # Add the image to face samples
                faceSamples.append(img_numpy[y:y+h,x:x+w])

                # Add the ID to IDs
                ids.append(id)
                #names.append(name)

    # Passing the face array and IDs array
    return faceSamples,ids

# Getting the faces and IDs
faces,ids = getImagesAndLabels('training_data')



# Training the model using the faces and IDs
recognizer.train(faces, np.array(ids))

print("Trained Successfully !!")

# Saving the model into s_model.yml
assure_path_exists('saved_model/')
recognizer.write('saved_model/s_model.yml')
