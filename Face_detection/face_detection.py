import cv2
import pygame
# import os
# import face_recognition

def faceDet():

    face_cascade = cv2.CascadeClassifier('/home/pranil/.local/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/home/pranil/.local/lib/python3.10/site-packages/cv2/data/haarcascade_eye.xml')
    cap = cv2.VideoCapture(0)       #when cv2.VideoCapture(0) := 0 is set as an argument, it indicates that OpencV should use the default camera device 
    
    while True:
        ret, img = cap.read()       #reads frames from a camera
        if not ret:
            break
        
        #cvtColor converts the rgb to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #img and gray are numPy array i.e a multi dimensional array: matrix

        #detectMultiScale returns a list of rectanges representing the detected faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        pygame.mixer.init()
        sound = pygame.mixer.Sound('/home/pranil/pythonProjects/openCV/Face_detection/beep-01a.wav')

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,0), 2)      #x, y coordinates of top left rectange on the face w and h are width , 225, 225, 0 color of rectange, with thickness 2
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 127, 255), 2)

            sound.play()

        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

