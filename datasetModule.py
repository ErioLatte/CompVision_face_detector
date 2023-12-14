import os
import cv2 as cv
from sklearn.model_selection import train_test_split


# get the image path & its label
def getDatasets(datasets_path):
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []
    for idx, classes in enumerate(os.listdir(datasets_path)):
        # join path
        temp_path = os.path.join(datasets_path, classes)
        # temporary array to be split each loop
        temp_train = []
        temp_label = []
        # load image from each class
        for image in os.listdir(temp_path):
            image_path = os.path.join(temp_path, image)
            temp_train.append(image_path)
            temp_label.append(idx)
        # take 25% of image (from each class) to test
        x_train, x_test, y_train, y_test = train_test_split(temp_train, temp_label, test_size=0.25, random_state=82)
        train_images.extend(x_train)
        train_labels.extend(y_train)
        test_images.extend(x_test)
        test_labels.extend(y_test)

    return train_images, test_images, train_labels, test_labels

# error handling if xml file not found
def getClassifier(cascade):
    face_cascade = None
    if os.path.exists(cascade):
        face_cascade = cv.CascadeClassifier(cascade)
    else:
        print("xml file for cascade not found, exiting program")
        exit()
    return face_cascade

# will crop image into face only & resize it
def cropImage(cascade, images, labels=None, width=640, height=480):
    # to contain image that detected only 1 face
    detected_images = []
    detected_labels = []
    
    # get classifier
    classifier = getClassifier(cascade)
    count = 0

    
    # loop through the image list
    for img, lbl in zip(images, labels):
        gray_image = cv.imread(img, 0)
        # find face
        faces = classifier.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_DO_ROUGH_SEARCH)
        # if only one face detected 
        if len(faces)==1:
             count+=1
             # take the face coordinate
             for (x, y, w, h) in faces:
                # crop the face
                face_image = gray_image[y:y+h, x:x+w] 
                # resize the face -> normalize
                face_image = cv.resize(face_image, (width, height), interpolation = cv.INTER_CUBIC)

                face_image = face_image.astype('float32')
                face_image /= 255.0

                detected_images.append(face_image)
                detected_labels.append(lbl)
    # print(count)
    
    return detected_images, detected_labels

