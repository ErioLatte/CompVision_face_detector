import cv2 as cv
import os
import numpy as np



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

def predict(path, cascade, model):
    if not os.path.isfile(path):
        print("path doesnt exist")
        return
    # face_recognizer = cv.face.LBPHFaceRecognizer_create()
    # face_recognizer.read(model)
    # image = 0
    # count = 0
    # for img, lbl in zip(test_images, test_labels):
    #     result, _ = face_recognizer.predict(img)
    #     if result == lbl:
    #         count+=1
    # accuracy = count/len(test_labels)
    # print(f"Average Acccuracy: {count} / {len(test_labels)}")
    print("predicting")

# MODEL_PATH = ".\\models\\trainedModel.yml"
# dataset = "C:\\Users\\erioy\\OneDrive\\Documents\\GitHub\\CompVision_face_detector\\Dataset\\cristiano_ronaldo\\1.jpg"
# predict_path = input("Input absolute path for image to predict >> ")
# predict(MODEL_PATH, dataset)
