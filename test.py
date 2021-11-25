import cv2
import time
import easyocr
import string
from Plate import plate_detection, allow_list, reader


def is_LP_square(image):
    H, W = image.shape[:2]
    if W / H > 2.5:
        return False
    return True


def CCA(image):
    H, W = image.shape[:2]
    output = cv2.connectedComponentsWithStats(
        image, 2, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    bboxes = []
    for i in range(2, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        if H / h > 10:
            continue
        bboxes.append([x, y, w, h])
    return bboxes


def bboxes_square_LP(image):
    bboxes = CCA(image)
    y_mean = sum([i[1] for i in bboxes]) / len(bboxes)
    line1 = []
    line2 = []
    for bbox in bboxes:
        x, y, w, h = bbox
        if y < y_mean:
            line1.append([x, y, w, h])
        else:
            line2.append([x, y, w, h])
    line1 = sorted(line1)
    line2 = sorted(line2)
    bboxes_LP = line1 + line2
    return bboxes_LP


def bboxes_rec_LP(image):
    bboxes = CCA(image)
    return sorted(bboxes)


def process_LP(image, allow_list):
    lp_text = ""
    if is_LP_square(image):
        LP = bboxes_square_LP(image)
    else:
        LP = bboxes_rec_LP(image)

    for x, y, w, h in LP:
        roi = image[y - 2:y + h + 2, x - 2:x + w + 2]
        cv2.rectangle(crop, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 2)
        try:
            (_, t, conf) = reader.readtext(roi, allowlist=allow_list, decoder="greedy", min_size=5,
                                           text_threshold=0.4, low_text=0.2, link_threshold=0.2, mag_ratio=3)[0]
            if conf > 0.1:
                lp_text += t
        except:
            pass
    cv2.imshow("Roi", crop)
    cv2.waitKey()
    return image, lp_text


image_ori = cv2.imread("ImageStorage/images.jpg")
# H, W = image.shape[:2]
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Ap dung threshold de phan tach so va nen
# binary = cv2.threshold(gray, 0, 255,
#                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#
# # Segment kí tự
# kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# thre_mor = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel3)
status, crop, thres_mor = plate_detection(image_ori)

t = time.time()
thres_mor, lp_text = process_LP(thres_mor, allow_list)
print("Time taken:", time.time() - t)
print(lp_text)
cv2.imshow("Image", image_ori)
cv2.imshow("Thresh", thres_mor)
cv2.waitKey()
