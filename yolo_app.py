import cv2
import argparse
import numpy as np
import os
import time
import tensorflow
from tensorflow.keras.preprocessing.image import load_img
import streamlit as st
from keras.models import load_model

training_weight_url="https://drive.google.com/uc?id=1XPppB5-oC-9OR4hV9okFZynlSceQ84we"

url = 'https://pjreddie.com/media/files/yolov3.weights'
if not os.path.exists("yolo-coco/yolov3.weights"):
    print("weights_path_insde")
    response = requests.get(url)
    if response.status_code == 200:
        # Save the file to disk
        with open('yolo-coco/yolov3.weights', 'wb') as f:
            f.write(response.content)
if not os.path.exists("yolo-coco/inceptionv3.h5"):
    output='yolo-coco/inceptionv3.h5'
    gdown.download(training_weight_url,output,quiet=False)

model = load_model('yolo-coco/inceptionv3.h5')

#args = ap.parse_args()
# Define Streamlit app
#model = load_model('/Users/shikha/PycharmProjects/pythonProject/human-detection/model_inception.h5')
def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers
def app():
    st.title("Person Detection using YOLO")
    st.write("Upload an image or select webcam to detect persons")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            # Save uploaded image to temporary file
            if not os.path.exists("temp"):
                os.makedirs("temp")
            temp_file = os.path.join("temp", uploaded_file.name)
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getvalue())

                # Load image and convert to grayscale
                img = cv2.imread(temp_file)
                #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Perform person detection on uploaded image
            detect_persons1(img)

        except Exception as e:
            st.write(f"Error processing image: {str(e)}")

    # Select webcam
    if st.button("Use Webcam"):
        try:
            # Open webcam for video capture
            stop = st.button("Stop")  # Button to stop capturing photos
            cap = cv2.VideoCapture(0)
            frame_number=0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                print(frame.shape)
                frame_number += 1
                # Perform person detection on webcam frame
                detect_persons1(frame)

                # Display webcam frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame, channels="RGB", use_column_width=True)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if stop:
                    break
                if frame_number % 300 == 0:  # Capture interval fixed at 10 seconds (10 * 30 frames)
                    time.sleep(1)

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            st.write(f"Error capturing webcam video: {str(e)}")

    st.write("Finished processing.")

# Perform person detection
def detect_persons(image):
    print(image.shape,args.weights,args.config)
    # Load classes
    classes = None
    with open('yolo-coco/yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Generate random colors for each class
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # Load YOLO model
    net = cv2.dnn.readNet('yolo-coco/yolov3.weights','yolo-coco/yolov3.cfg')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    # Apply non-maximum suppression to remove overlapping detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw bounding boxes around detected persons
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        color = COLORS[class_ids[i]]
        cv2.rectangle(image, (round(x), round(y)), (round(x + w), round(y + h)), color, 2)
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(image, label, (round(x), round(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert image back to RGB for displaying in Streamlit
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display image with bounding boxes in Streamlit
    st.image(image, channels="RGB", use_column_width=True)


def detect_persons1(image):
    #image = cv2.imread(image)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
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
        color = COLORS[class_ids[i]]
        cv2.rectangle(image, (round(x), round(y)), (round(x + w), round(y + h)), color, 2)
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(image, label, (round(x), round(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cropped_image = image[round(y):round(y + h), round(x):round(x + w)]
        cv2.imwrite("output_img.png", cropped_image)
        img_path = '/Users/shikha/PycharmProjects/pythonProject/human-detection/output_img.png'
        img = load_img(img_path, target_size=(224, 224))
        img_array = tensorflow.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        # Make a prediction
        preds = model.predict(img_array)

        print("pred",preds)
        # Decode the prediction
        classes = ['swiggy', 'zomato', ]
        pred_class = classes[np.argmax(preds)]

        print('The predicted class is:', pred_class)
        st.text(pred_class)



    # Convert image back to RGB for displaying in Streamlit
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display image with bounding boxes in Streamlit
    st.image(image, channels="RGB", use_column_width=True)
app()


# streamlit run yolo_app.py -- --config /Users/shikha/PycharmProjects/pythonProject/human-detection/yolo-coco/yolov3.cfg --weights /Users/shikha/PycharmProjects/pythonProject/human-detection/yolo-coco/yolov3.weights --classes /Users/shikha/PycharmProjects/pythonProject/human-detection/yolo-coco/yolov3.txt
