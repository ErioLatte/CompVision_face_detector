import os
from datasetModule import getDatasets, cropImage
from trainModule import predict, showImage, trainImage

CASCADE_PATH = ".\\models\\haarcascade_frontalface_default.xml"
PATH = ".\\Dataset"
MODEL_PATH = ".\\models\\trainedModel.yml"
train_images, test_images, train_labels, test_labels = getDatasets(PATH)
# define image size
width = 640
height = 480 


def mainMenu():
    print("Footbal Player Face Recognition")
    print("1. Train and Test Model\n2. Predict\n3. Exit")
    user_choice = input(">> ")
    return user_choice

user_choice = 0
while(user_choice != '3'):
    os.system('cls')
    user_choice = mainMenu()
    
    if user_choice == '1':
        print("Trainin and Testing")
        train_images, train_labels = cropImage(CASCADE_PATH, train_images, train_labels, width, height)
        test_images, test_labels = cropImage(CASCADE_PATH, test_images, test_labels, width, height)
        trainImage(train_images, train_labels, test_images, test_labels, MODEL_PATH)
        print("Training and Testing Finished")
    elif user_choice == '2':
        predict_path = input("Input absolute path for image to predict >> ")
        predict(predict_path, CASCADE_PATH, MODEL_PATH, width, height)
    elif user_choice == '3':
        print("EXIT")
    else:
        print("please enter number between 1-3!")
    
    input("press enter to continue...")