# TODO:太久了不知道这个脚本干什么的

import cv2
import cv2 as cv
from PIL import Image
import numpy as np
from numpy import split
import torch
import os


def xywhn2xyxy(x, w, h, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = w * (x[0] - x[2] / 2) + padw  # top left x
    y[1] = h * (x[1] - x[3] / 2) + padh  # top left y
    y[2] = w * (x[0] + x[2] / 2) + padw  # bottom right x
    y[3] = h * (x[1] + x[3] / 2) + padh  # bottom right y
    return y


def before_guiyi(x, w, h):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = int(w * x[0])
    y[1] = int(h * x[1])
    y = y.astype(np.int64)
    return y


def changelabel(label_path, imwrite_path):
    files = os.listdir(label_path)
    for file in files:
        f = open(label_path + file)
        dataset = f.readlines()
        save = []
        # print(dataset)
        for line in dataset:
            tempdata1 = line.split()
            tempdata2 = np.array([float(x) for x in tempdata1[1:]])
            points = np.split(tempdata2, len(tempdata2) // 2)
            x = str((points[0][0] + points[2][0]) / 2)
            y = str((points[0][1] + points[2][1]) / 2)
            w = str((max(points[2][0], points[0][0]) - min(points[2][0], points[0][0])) / 2)
            h = str((max(points[0][1], points[2][1]) - min(points[0][1], points[2][1])) / 2)
            box = [x, y, w, h]

            save_data = [tempdata1[0]] + box + tempdata1[1:]
            # print(save_data)
            save.append(save_data)
        f.close()
        # save=sum(save, [])
        f1 = open(imwrite_path + file, 'w')
        for sav in save:
            for sa in sav:
                f1.write(sa)
                f1.write(' ')
            f1.write("\n")
        f1.close()
        save = []


image_path = r'C:\Users\59781\Downloads\Compressed\yolov5-Heading-master\dataset\train\images/'
label_path = r'/dataset/train/labels/'
imwrite_path = r'C:\Users\59781\Downloads\Compressed\yolov5-Heading-master\dataset\train\labels/'

changelabel(label_path, imwrite_path)

h = 1280
w = 1024
img_PIL = cv2.imread(r'C:\Users\59781\Downloads\Compressed\yolov5-Heading-master\dataset\train\images/515.jpg')
f = open(r'C:\Users\59781\Downloads\Compressed\yolov5-Heading-master\dataset\train\labels/515.labels', 'r')

dataset = f.readlines()
for lines in dataset:
    temp = lines.split()

    _temp = np.array([float(x) for x in temp[1:]])

    points = np.split(_temp, len(_temp) // 2)

    print(points)

    x, y = before_guiyi(points[0], h, w)
    w1, h1 = before_guiyi(points[1], h, w)

    cv2.circle(img_PIL, (x, y), 1, (255, 0, 255))
    rect = cv2.rectangle(img_PIL, (x - w1, y + h1), (x + w1, y - h1), (255, 0, 255))
#     x1, y1, w, h = rect
#     print(x1, y1, w, h)
#
cv2.imshow("src", img_PIL)
cv2.waitKey(0)
