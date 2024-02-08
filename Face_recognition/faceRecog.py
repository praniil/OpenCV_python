import cv2
import os
import face_recognition

def faceRec():
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir('/home/pranil/pythonProjects/openCV/known_people'):
        image = face_recognition.load_image_file(os.path.join('/home/pranil/pythonProjects/openCV/known_people', file_name))
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(file_name)[0])

    cap = cv2.VideoCapture(0)       #when cv2.VideoCapture(0) := 0 is set as an argument, it indicates that OpencV should use the default camera device 
    
    while True:
        ret, img = cap.read()       #reads frames from a camera
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    #face recognition library uses RGB images

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                index = matches.index(True)
                name = known_face_names[index]
            
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left+6, bottom-6), font, 0.5, (255,255,255), 1)

        cv2.imshow('Face recognition', img)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
            
