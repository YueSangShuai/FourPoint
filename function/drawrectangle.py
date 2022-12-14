# TODO:用于把上交格式的数据集转成yoloface,并且处理了数据类别不连续的情况并把重新处理后的类别输出一个txt文件方便对照
import cv2
import cv2 as cv
from PIL import Image
import numpy as np
from numpy import split
import torch
import os
from count_classes import *


def cross_point(line1, line2):  # 计算交点函数
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


def xywhn2xyxy(x, w, h):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = int(w * (x[0] - x[2] / 2))  # top left x
    y[1] = int(h * (x[1] - x[3] / 2))  # top left y
    y[2] = int(w * (x[0] + x[2] / 2))  # bottom right x
    y[3] = int(h * (x[1] + x[3] / 2))  # bottom right y
    return y


def before_guiyi(x, w, h):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = int(w * x[0])
    y[1] = int(h * x[1])
    y = y.astype(np.int64)
    return y


def after_guiyi(x, w, h):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = w / x[0]
    y[1] = h / x[1]
    y = y.astype(np.int64)
    return y


def changelabel(label_path, imwrite_path, innumber, outnumber):
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
            line1 = [points[0][0], points[0][1], points[2][0], points[2][1]]
            line2 = [points[1][0], points[1][1], points[3][0], points[3][1]]
            x, y = cross_point(line1, line2)
            x = str(x)
            y = str(y)
            # x = str((points[0][0] + points[2][0]) / 2)
            # y = str((points[0][1] + points[2][1]) / 2)
            w = str((max(points[0][0], points[2][0]) - min(points[0][0], points[2][0])) * 1.5)
            h = str((max(points[1][1], points[2][1]) - min(points[1][1], points[2][1])))

            box = [x, y, w, w]
            point1 = tempdata1[1:3]
            point2 = tempdata1[3:5]
            point3 = tempdata1[5:7]
            point4 = tempdata1[7:9]

            point = [0, 0]
            save_data = [str(tempdata1[0])] + box + tempdata1[1:]
            # for i in range(13):
            #     save_data += [str(point[0])]
            #     save_data += [str(point[1])]
            save.append(save_data)

            # for cls in dict:
            #     if cls[0] == int(tempdata1[0]):
            #         save_data = [str(cls[1])] + box + point1 + point2 + point4 + point3 + [x, y]
            #         save.append(save_data)
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


if __name__ == '__main__':
    # image_path = r'..\dataset\test\images/'
    label_path = r'/media/yuesang/G/Robotmaster/dataset/data/labels/temp/'
    imwrite_path = r'/media/yuesang/G/Robotmaster/dataset/data/labels/train/'
    changelabel(label_path, imwrite_path, 1, 1)

    # img_PIL = cv2.imread(r'E:\Robotmaster\dataset\mydata\images\train\9.jpg')
    # f = open(r'E:\Robotmaster\dataset\mydata\labels\train\9.txt', 'r')
    # h = 720
    # w = 360
    #
    # dataset = f.readlines()
    # for lines in dataset:
    #     temp = lines.split()
    #
    #     _temp = np.array([float(x) for x in temp[1:]])
    #
    #     points = np.split(_temp, len(_temp) // 2)
    #
    #     print(points)
    #
    #     x, y = before_guiyi(points[0], h, w)
    #     w1, h1 = before_guiyi(points[1], h, w)
    #
    #     cv2.circle(img_PIL, (x, y), 1, (255, 0, 255))
    #     rect = cv2.rectangle(img_PIL, (x - w1, y + h1), (x + w1, y - h1), (255, 0, 255))
    # #     x1, y1, w, h = rect
    # #     print(x1, y1, w, h)
    # #
    # cv2.imshow("src", img_PIL)
    # cv2.waitKey(0)
