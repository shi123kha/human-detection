import cv2
import argparse
import numpy as np
import os

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True,
                help='path to input image folder')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
ap.add_argument('-o', '--output', required=True,
                help='path to output folder for cropped images')
args = ap.parse_args()


def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# Load classes
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Load YOLO model
net = cv2.dnn.readNet(args.weights, args.config)

# Process images in input folder
for filename in os.listdir(args.input):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        try:
            image_path = os.path.join(args.input, filename)
            image = cv2.imread(image_path)
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
                draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

                # Save cropped image
                cropped_image = image[round(y):round(y + h), round(x):round(x + w)]
                output_filename = os.path.splitext(filename)[0] + '_' + str(i) + '.jpg'
                output_path = os.path.join(args.output, output_filename)
                cv2.imwrite(output_path, cropped_image)
                print(f'Saved cropped image: {output_path}')

                # Show the image with detections
                #cv2.imshow('Object Detection', image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

        except Exception as e:
                print(f'xError processing image {filename}: {str(e)}')

print('Finished processing all images.')

#python yolo_video.py   --config /Users/shikha/PycharmProjects/pythonProject/human-detection/yolo-coco/yolov3.cfg --weights /Users/shikha/PycharmProjects/pythonProject/human-detection/yolo-coco/yolov3.weights --classes /Users/shikha/PycharmProjects/pythonProject/human-detection/yolo-coco/yolov3.txt --input /Users/shikha/Desktop/person_detection/Dunzo --output /Users/shikha/Desktop/person_detection/Dunzo/
