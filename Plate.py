import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single
import os
import easyocr
import string

reader = easyocr.Reader(lang_list=["en"])

wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

digit_w = 30  # Kich thuoc ki tu
digit_h = 60  # Kich thuoc ki tu

char_list = '0123456789ABCDEFGHKLMNPRSTUVXYZ'


def fine_tune(lp):
    newString = ""
    index = 0
    if len(lp) < 2:
        return lp
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
            if lp[i].isdigit():
                index = i
    if index > 2 and newString[0] == "1":
        newString = newString[1:]
    newString = list(newString)
    if newString[2] == "6":
        newString[2] = "G"
    elif newString[2] == "0":
        newString[2] = "D"
    elif newString[2] == "4":
        newString[2] = "A"
    elif newString[2] == "7":
        newString[2] = "Y"
    result = "".join(newString)
    if len(result) > 8:
        result = result[:-1]
    return result


def plate_detection(Ivehicle):
    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    try:
        _, LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)
    except:
        return False, False

    # Cau hinh tham so cho model SVM

    if not (len(LpImg)):
        return False, False
        # Chuyen doi anh bien so
    LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

    roi = LpImg[0]

    return True, roi


def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ap dung threshold de phan tach so va nen
    binary = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Segment kí tự
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel3)
    text = reader.readtext(thre_mor, detail=0, allowlist=string.ascii_uppercase + string.digits)

    return thre_mor, text


def sort_contours(cnts):
    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


def check_contours(x, y, w, h, cnts):
    x_center = x + w // 2
    y_center = y + h // 2
    for x1, y1, w1, h1 in cnts:
        if (x1 < x_center < x1 + w1) and (y1 < y_center < y1 + h1):
            return False
    return True


def process_image_chracter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ap dung threshold de phan tach so va nen
    binary = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Segment kí tự
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel3)

    cont, _ = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lp_text = ""
    height, width = thre_mor.shape
    ls_cnts = [[0, 0, 0, 0]]
    count = 0
    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        raito = h / w
        if x < 5 or y + h > height - 5:
            continue
        if height / h > 2.5 or height / h < 1.1:
            continue
        if raito < 1:
            continue
        if not check_contours(x, y, w, h, ls_cnts):
            continue
        ls_cnts.append([x, y, w, h])
        cv2.rectangle(image, (x - 2, y - 4), (x + w + 2, y + h + 4), (0, 0, 255), 0)
        roi = thre_mor[y - 4:y + h + 4, x - 2:x + w + 2]
        count += 1
        if count == 3:
            allow_list = string.ascii_uppercase
        else:
            allow_list = string.digits
        # roi = cv2.resize(roi, dsize=None, fx=3, fy=3)
        try:
            (_, t, conf) = reader.readtext(roi, allowlist=allow_list, decoder="greedy", min_size=5,
                                text_threshold=0.4, low_text=0.2, link_threshold=0.2, mag_ratio=3)[0]
            print(count, t, conf)
            if conf > 0.1:
                lp_text += t
        except:
            pass

    # cv2.imshow("ROI", roi)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    print("-" * 50)
    return thre_mor, lp_text


if __name__ == '__main__':
    data_path = r"D:\Lab IC\LP\data17-7\data17-7-2021"

    for path in os.listdir(data_path):
        print(path)
        image = cv2.imread(f"{data_path}/{path}")
        LP_image = plate_detection(image)
        cv2.imshow("LP", LP_image)
        cv2.imshow("Image", image)
        cv2.waitKey()
