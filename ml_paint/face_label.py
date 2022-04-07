from imutils import paths
import face_recognition
import pickle
import cv2
import os
import imutils
import time
import numpy as np

baseDir = "/home/sabeiro/lav/tmp/gan/"
baseDir = "/home/sabeiro/lav/src/gen/"
wrtDir = "/home/sabeiro/tmp/pers/"
imgDir = "/home/sabeiro/Pictures/pers/"

imagePaths = list(paths.list_images(wrtDir+"img"))
colorL = ['red','blue','green']
encL = []
for (i, imagePath) in enumerate(imagePaths):
        name = imagePath.split(os.path.sep)[-1].split("-")[0]
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb,model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)
        r, g, b = np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2])
        bright = np.mean([r,g,b])
        color = colorL[np.argmax([r,g,b])]
        for enc in encodings:
                d = {"enconding":enc,"name":name,"red":r,"blue":b,"green":g,"bright":bright
                     ,"color":color,"fileName":imagePath}
                encL.append(d)

labelL = {}
d = encL[0]
for k in d.keys(): labelL[k] = [v[k] for v in encL]
f = open(wrtDir+"train/face_enc.pkl", "wb")
f.write(pickle.dumps(labelL))
f.close()

