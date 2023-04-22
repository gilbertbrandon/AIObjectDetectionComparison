#IMPORT REQUIRED DEPENDCIES
import cv2 as cv
import numpy as np


#OPENS WEBCAM FRAME
##cap = cv.VideoCapture(0)
#OPENS VIDEO FRAME
cap = cv.VideoCapture('dashcam2.mp4')

#VARIABLES
whT = 320
confThreshold = 0.5
nmsThreshold = 0.2

#### LOAD MODEL CLASSIFIERS
## Coco Names = TXT FILE THAT CONTAINS OBJECT CLASSES
classesFile = "coco.names"
classNames = []

#Extracts CLASS NAMES with new line
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

## Model Files
modelConfiguration = "yolov3.cfg.txt"
modelWeights = "yolov3.weights"

## CREATE NETWORKS THAT TAKES IN CONFIG AND WEIGHT FILES
net = cv.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
## SETS OPENCV AS PREFERED BACKEND
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

##METHOD THAT FINDS THE OBJECT IN FRAME
def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    ##CLASSIFERS IDS
    classIds = []
    ##CONFIDENCE VALUES
    confs = []
    for output in outputs:
        for det in output:
            ##REMOVES THE FIRST FIVE ELEMENTS
            scores = det[5:]
            ##FIND INDEX OF MAX VALUE
            classId = np.argmax(scores)
            ##GET CONFIDENCE VALUE
            confidence = scores[classId]
            ##FILTERS THE OBJECTS
            if confidence > confThreshold:
                ##GET PIXEL VALUES OF W,H
                w, h = int(det[2] * wT), int(det[3] * hT)
                ##GET PIXEL VALUE OF X,Y
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                ##ADDS BOUNDING BOX TO X,Y,W,H
                bbox.append([x, y, w, h])
                ##ADDS CLASSIDS
                classIds.append(classId)
                ##ADDS CONFIDENCE VALUE
                confs.append(float(confidence))
    ##OUTPUTS THE FOLLOWING BASED ON THE INDICES
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    ##LOOPS OVER THE INDICES AND DRAWS THE BOUNDING BOX
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        ##CONFIGURES THE SIZE OF BOUNDING BOX
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        ##CONFIGURES AND SETS THE CLASSIDS TO DISPLAY CLASS NAMES AND APPENDS CONFIDENCE
        cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                   (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


#WILL DISPLAY IMAGE AND RETURNS WHEATHER RETRIEVEAL WAS SUCCESSFUL OR NOT
while True:
    success, img = cap.read()

    ##CONVERTS IMAGE TO BLOB AND DIVIDES INTO 255
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    ##SET BLOB AS OUR NETWORK INPUT
    net.setInput(blob)
    ##GET LAYER NAMES FOR EACH PREDICTION LAYER
    layersNames = net.getLayerNames()
    ##EXTRACT OUTPUT LAYERS
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    ##GET THE 3 OUTPUT NAMES
    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    #DISPLAYS IMAGE
    cv.imshow('Image', img)
    #WILL DISPLAY UNTIL KEY PRESS
    cv.waitKey(1)