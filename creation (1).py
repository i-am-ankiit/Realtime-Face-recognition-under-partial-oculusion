import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import csv
import datetime
import time
from tensorflow.keras.models import load_model
from keras.preprocessing import image                  
from scipy.spatial import distance as dist
import argparse
import imutils
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

##year = input("enter the year")
##print("year:" +year+ ".")
##
##dept = input("enter the department")
##print("dept:" + dept + ".")
##
##sem = input("enter the semester")
##print("sem:" + sem + ".")
##
##subj = input("enter the subject")
##print("subject:" + subj + ".")

name = input("enter the name")
print("name:" + name + ".")

 # Replace with your desired folder name
path = "C:/Users/ak398/OneDrive/Desktop/face project/data"+"/"+name  # Replace with the directory path where you want to create the folder

if not os.path.exists(path):
    os.makedirs(path)
    print("Folder created successfully.")
else:
    print("Folder already exists.")


# Define the path where you want to store the captured images
path = path

# Define the number of images to capture
num_images = 40

# Initialize the camera
camera = cv2.VideoCapture(0)

# Define a counter variable to keep track of the number of images captured
counter = 0

while counter < num_images:
    # Capture an image from the camera
    ret, frame = camera.read()

    # Check if the image was successfully captured
    if ret:
        # Define the filename for the image
        filename = f"image{counter}.jpg"
        faces=detector.detect_faces(frame)
        for result in faces:
            x1, y1, w, h = result['box']
            x2=x1+w
            y2=y1+h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
##            face_cordinates.append((x1, y1, w, h))
            
            face_image =frame[y1:y1+h,x1:x1+w]
##            cv2.imwrite('./Images/'+str(cnt)+'.jpg',face_image)
            file_path = os.path.join(path, filename)
##            cv2.imwrite(file_path, frame)
            cv2.imwrite(file_path,face_image)

        # Save the image to the specified path
##        file_path = os.path.join(path, filename)
##        cv2.imwrite(file_path, frame)
        #cv2.imwrite(path +filename, frame)

        # Increment the counter
        counter += 1

    # Display the image on screen
    cv2.imshow('Image', frame)

    # Wait for a key press to continue or quit
    key = cv2.waitKey(1)

    # Press 'q' to quit the program
    if key == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()


