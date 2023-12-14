import cv2 as cv
import os
import numpy as np
import math
from datasetModule import getClassifier


def getClassName(index):
    classes = os.listdir(".\\Dataset")
    return classes[index]

def showImage(images, labels):
    for x, y in zip(images, labels):
        cv.imshow("asd", x)
        cv.waitKey(0)
        cv.destroyAllWindows()

def trainImage(train_images, train_labels, test_images, test_labels, filepath):
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(train_images, np.array(train_labels))
    
    count = 0
    for img, lbl in zip(test_images, test_labels):
        result, _ = face_recognizer.predict(img)
        if result == lbl:
            count+=1
    accuracy = count/len(test_labels)
    print(f"Average Acccuracy: {accuracy*100}")
    
    face_recognizer.save(filepath)
    return

def predict(path, cascade, model, width=640, height=480):
    # if path error
    if not os.path.isfile(path):
        print("path doesnt exist")
        return
    # if model not exist
    if not os.path.isfile(model):
        print("model not found!")
        return
    
    # load model
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model)

    # load cascade
    classifier = getClassifier(cascade)
    
    gray_image = cv.imread(path, 0)
    faces = classifier.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_DO_ROUGH_SEARCH)
    # if face is detected
    if len(faces)>=1:
        for (x, y, w, h) in faces:
            # crop the face
            face_image = gray_image[y:y+h, x:x+w] 
                    # resize the face -> normalize
            face_image = cv.resize(face_image, (width, height), interpolation = cv.INTER_CUBIC)

            face_image = face_image.astype('float32')
            face_image /= 255.0

            #predict
            result, confidence = face_recognizer.predict(face_image)
            
            confidence = 100 - math.floor(confidence*100)/100
            # draw bounding box
            color_image = cv.imread(path)
            cv.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0))
            confidence_text = f'{confidence}%'
            name_text = f'{getClassName(result)}'
            cv.putText(color_image, confidence_text, (x+w+1, y+20), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
            cv.putText(color_image, name_text, (x-5, y-10), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
            cv.imshow("result", color_image)
            cv.waitKey(0)
            cv.destroyAllWindows()

            break
    else:
        print("face not detected!")
    return

