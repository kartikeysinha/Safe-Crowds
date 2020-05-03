import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import animation

def get_data(path):
    labelsPath = "./coco.names"
    weightsPath = "./yolov3.weights"
    configPath = "./yolov3.cfg"
    cap = cv2.VideoCapture(path)
    hasFrame, frame = cap.read()
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    alert = 0
    count = 0
    current_frame = 0
    vals = []
    while cv2.waitKey(1) < 0 and cap.isOpened():
        current_frame += 1
        current_count = 0
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.resize(image, (640, 360))
        (H, W) = image.shape[:2]
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 300.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.1 and classID == 0:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        ind = []
        for i in range(0, len(classIDs)):
            if (classIDs[i] == 0):
                ind.append(i)
        coordinates = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                coordinates.append((x,y,w,h))
        violators = set()
        for i in range(0, len(coordinates) - 1):
            for k in range(1, len(coordinates)):
                if (k == i):
                    continue
                else:
                    x1,y1,w1,h1 = coordinates[i]
                    x2,y2,w2,h2 = coordinates[k]
                    x_dist = (x1 - x2)
                    y_dist = (y1 - y2)
                    d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                    if (d <= 40):
                        alert += 1
                        current_count += 1
                        # image = cv2.line(image,(x1+w1//2, y1+h1//2),(x2+w2//2,y2+h2//2),(0, 255, 255),1)
                        if i not in violators:
                            image = cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(0,0,255),1)
                            cv2.putText(image, 'Caution', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                            violators.add(i)
                        if k not in violators:
                            image = cv2.rectangle(image, (x2, y2), (x2+w2,y2+h2),(0,0,255),1)
                            cv2.putText(image, 'Caution', (x2, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                            violators.add(k)
        for i in range(len(coordinates)):
            if i not in violators:
                x,y,w,h = coordinates[i]
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 1)
                cv2.putText(image, 'OK', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)
        cv2.imshow('Social Distance Detector',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        vals.append(current_count)
    cap.release()
    cv2.destroyAllWindows()
    return vals,alert

