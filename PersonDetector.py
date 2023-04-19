import os
import cv2
import requests
import gdown
import numpy as np
import streamlit as st
import tensorflow
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img

class PersonDetector:

    def __init__(self, weights_url='https://pjreddie.com/media/files/yolov3.weights',
                 model_url='https://drive.google.com/uc?id=1XPppB5-oC-9OR4hV9okFZynlSceQ84we',
                 model_path='yolo-coco/inceptionv3.h5', weights_path='yolo-coco/yolov3.weights',
                 class_path='yolo-coco/yolov3.txt'):

        self.weights_url = weights_url
        self.model_url = model_url
        self.model_path = model_path
        self.weights_path = weights_path
        self.class_path = class_path

        self.__download_weights()
        self.__download_model()

        self.model = load_model(self.model_path)
        with open(self.class_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def __download_weights(self):
        if not os.path.exists(self.weights_path):
            response = requests.get(self.weights_url)
            if response.status_code == 200:
                with open(self.weights_path, 'wb') as f:
                    f.write(response.content)

    def __download_model(self):
        if not os.path.exists(self.model_path):
            output = self.model_path
            gdown.download(self.model_url, output, quiet=False)

    def __get_output_layers(self, net):
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def detect_persons(self, image):
        outs = self.model.predict(image)
        confidences = []
        boxes = []
        class_ids = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > conf_threshold:
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    w = int(detection[2] * image.shape[1])
                    h = int(detection[3] * image.shape[0])
                    x = center_x - w // 2
                    y = center_y - h // 2
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            try:
                box = boxes[i]
            except:
                i = i[0]
                box = boxes[i]

            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            color = self.COLORS[class_ids[i]]
            cv2.rectangle(image, (round(x), round(y)), (round(x + w), round(y + h)), color, 2)
            label = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(image, label, (round(x), round(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cropped_image = image[round(y):round(y + h), round(x):round(x + w)]
            cv2.imwrite("output_img.png", cropped_image)
            img_path = 'output_img.png'
            img = load_img(img_path, target_size=(224, 224))
            img_array = tensorflow.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            # Make a prediction
            preds = self.model.predict(img_array)

            print("pred",preds)
            class_labels = {0: 'Dunzo', 1: 'Ubereat', 2: 'others', 3: 'swiggy', 4: 'zomato'}
            predicted_class_index = np.argmax(preds)
            predicted_class_label = class_labels[predicted_class_index]
            st.text(predicted_class_label)

        # Convert image back to RGB for displaying in Streamlit
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)