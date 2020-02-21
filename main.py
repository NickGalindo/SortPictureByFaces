import face_recognition as fr
import os
import cv2
import numpy as np
from time import sleep
import shutil


class classifier():
    def __init__(self, dir=None):
        #dict with all encoded faces corresponding faceName:Encoding
        self.encoded = {}

        #check if directory parameter matched
        if dir == None:
            print("-----> NO DIRECTORY SPECIFIED. MANUAL CREATION REQUIRED <-----")
            return

        print("-----Loading face images-----")
        for file in os.listdir(dir):
            if file.endswith(".png") or file.endswith(".jpg"):
                print("Loading file -" + str(file) + "-")
                face = fr.load_image_file(str(dir)+"/"+file)
                faceEncoding = fr.face_encodings(face)[0]
                self.encoded[file.split(".")[0]] = faceEncoding

        print("-----Succesfully loaded face images-----")
        print("")
        print("-----Making parent directory-----")

        if os.path.exists("sorted"):
            print("Target directory -sorted- already exists")
        else:
            os.mkdir("sorted")
            print("Target directory -sorted- created")

        print("-----Making Child directories-----")

        for name in self.encoded.keys():
            if os.path.exists("sorted/"+name):
                print("Directory -sorted/"+str(name)+"- already exists")
            else:
                os.mkdir("sorted/"+str(name))
                print("Directory -sorted/"+str(name)+"- created succesfully")

        print("-----Successfully created child directories-----")

        if os.path.exists("sorted/unknown"):
            print("Directory -sorted/unkown- already exists")
        else:
            os.mkdir("sorted/unknown")
            print("Directory -sorted/unkown- created successfully")

        return

    def classify(self, dir=None):
        if dir == None:
            print("----- ABORTING -----     No directory specified.")
            return
        if not os.path.exists(dir):
            print("----- ABORTING -----     Specified directory doesn't exist.")
            return

        print("----- Initializing classficiation ----- ")

        compareFaces = list(self.encoded.values())
        faceNames = list(self.encoded.keys())

        for file in os.listdir(dir):
            if not file.endswith(".jpg") and not file.endswith(".png"):
                continue

            img = fr.load_image_file(dir+"/"+file)
            faceLocations = fr.face_locations(img)
            unknownfaceEncodings = fr.face_encodings(img, faceLocations)
            sorted = False

            for unknownFace in unknownfaceEncodings:
                matches = fr.compare_faces(compareFaces, unknownFace)

                faceDistances = fr.face_distance(compareFaces, unknownFace)
                bestMatchIndex = np.argmin(faceDistances)

                if matches[bestMatchIndex]:
                    shutil.move(os.getcwd()+"/"+str(dir)+"/"+str(file), os.getcwd()+"/sorted/"+str(faceNames[bestMatchIndex])+"/"+str(file))
                    print(os.getcwd()+"/"+str(dir)+"/"+str(file)+ " succesfully moved to "+os.getcwd()+"/sorted/"+str(faceNames[bestMatchIndex])+"/"+str(file))
                    sorted = True
                    break

            if not sorted:
                shutil.move(os.getcwd() + "/"+str(dir)+"/"+str(file), os.getcwd() +"/sorted/unknown")
                print(os.getcwd() + "/"+str(dir)+"/"+str(file)+" successfully moved to "+os.getcwd() +"/sorted/unknown")


if __name__ == "__main__":
    order = classifier("faces")
    order.classify("images")
