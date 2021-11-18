import time
import cv2
import numpy as np

image_path = r'C:\Users\Admin\Downloads\archive\images\Cars0.png'
config_path = 'yoloFile/yolov4_custom.cfg'
weight_path = 'yoloFile/LP.weights'
class_name_path = 'yoloFile/yolo.names'


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, index, x, y, x_plus_w, y_plus_h):
    color = (0, 255, 0)

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)


net = cv2.dnn.readNet(weight_path, config_path)
# image = cv2.imread(image_path)

with open(class_name_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

##########

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def detect_LP(image, conf_threshold, nms_threshold):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 1 / 255
    blob = cv2.dnn.blobFromImage(image, scale, (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    ###############################
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = conf_threshold
    nms_threshold = nms_threshold
    spot_list = []

    # Thực hiện xác định bằng HOG và SVM

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            # if class_id in [2, 5, 7]:
            confidence = scores[class_id]
            if confidence > conf_threshold:
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

    for index, i in enumerate(indices):
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], index + 1, None, round(x), round(y), round(x + w), round(y + h))
        spot_list.append([x, y, w, h])

    return image, spot_list
