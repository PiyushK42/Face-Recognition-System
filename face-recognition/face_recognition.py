# We will import openCV library for image processing, opening the webcam etc
#Os is required for managing files like directories
#Numpy is basically used for matrix operations
#PIL is Python Image Library
import cv2
import numpy as np
import os 
import xlwrite
import openpyxl
import xlwt 
from xlwt import Workbook 
import csv
import pandas as pd
import matplotlib.pyplot as plt

#Method for checking existence of path i.e the directory
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()
Ids=[]
dict={}
x=[]
y=[]

assure_path_exists("saved_model/")

# Load the  saved pre trained mode
recognizer.read('saved_model/s_model.yml')

# Load prebuilt classifier for Frontal Face detection
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);
'''names = {}
Ids= []
students = []
'''

# font style
font = cv2.FONT_HERSHEY_SIMPLEX


# Initialize and start the video frame capture from webcam
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,1200)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)

# Looping starts here
while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Getting all faces from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5) #default

    # For each face in faces, we will start predicting using pre trained model
    elem=0
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        #Set the name according to id
        ''' 
        if Id == 16115057:
           Id = "Piyush {0:.2f}%".format(round(100 - confidence, 2))
             #Put text describe who is in the picture
        elif Id == 16115019 :
            Id = "Devesh {0:.2f}%".format(round(100 - confidence, 2))
             #Put text describe who is in the picture
        elif Id == 16115023:
            Id = "Harsh {0:.2f}%".format(round(100 - confidence, 2))
        else:
            Id = "Unknown"
        '''
        # Set rectangle around face and name of the person

        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        if(confidence>100):
            cv2.putText(im, "Unknown", (x,y-40), font, 1, (255,255,255), 3)
        else:
            cv2.putText(im, str(Id)+"  "+str(round(confidence,2))+"%", (x,y-40), font, 1, (255,255,255), 3)
        

        ##for i in range(1, 90):
            ##filename = xlwrite.output('attendance','class1',i, 16115000 + i, 'no') 

        for i in range(1, 91):
            if((100-confidence)<50 and (100-confidence)>0): 
                if(Id == 16115000 + i):
                    if((str(Id)) not in dict):
                        filename = xlwrite.output('attendance','class1', i, 16115000 + i, 'YES')
                        dict[str(16115000 + i)]=str(16115000 + i) 

        for i in range(1, 91):
            if((str(16115000+i)) not in dict):
               filename = xlwrite.output('attendance','class1', i, 16115000 + i, 'NO') 

        '''            
            elif(Id==16115019):
                if ((str(Id)) not in dict):
                    filename =xlwrite.output('attendance', 'class1', 2, Id, 'yes');
                    dict[str(Id)] = str(Id);

            elif(Id==16115004):
                if ((str(Id)) not in dict):
                    filename =xlwrite.output('attendance', 'class1', 3, Id, 'yes');
                    dict[str(Id)] = str(Id);

            elif(Id==16115035):
                if ((str(Id)) not in dict):
                    filename =xlwrite.output('attendance', 'class1', 4, Id, 'yes');
                    dict[str(Id)] = str(Id);        
        '''

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 
    '''count=0
    i=0
    for item in y:
        count+=1
    while i<=count:
        x.append(i)
        i+=1
    plt.plot(x,y,'b')
    plt.xlabel('Inputs')
    plt.ylabel('100-Confidence')
    plt.show()'''

    # press q to close the program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        #update_Excel()
        break


'''
    if((100-confidence)<50 and (100-confidence)>0):
        Ids.append(Id)
        for item in Ids:
            if(str(Id) not in dict):
                filename=xlwrite.output('attendance','class1',2,str(Id),'yes')
                dict[str(Id)]=Id
'''
    


# Terminate video
cam.release()

# Close all windows
cv2.destroyAllWindows()
