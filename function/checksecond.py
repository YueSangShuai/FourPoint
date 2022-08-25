# TODO:用于判断生成的数据是否合法主要是用于把数据集中的点反归一化后重新展现在图片上
import os
import cv2
import numpy as np
import torch


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


image_path = r'E:\Robotmaster\2021-RMUC-0417-0916\images\\'
xml_path = r'E:\Robotmaster\2021-RMUC-0417-0916\labels\\'  # 数据原图路径

image_lst = os.listdir(image_path)
xml_lst = os.listdir(xml_path)
xml_filename = []
img_filename = []

for xml in xml_lst:
    xml = xml.split('.')
    xml_filename.append(xml[0])
for img in image_lst:
    img = img.split('.')
    img_filename.append(img[0])

for xml in xml_filename:

    xmlfilename = xml_path + "%s.txt" % xml
    imgfilename = image_path + "%s.jpg" % xml
    img = cv2.imread(imgfilename)
    w = img.shape[0]
    h = img.shape[1]
    f = open(xmlfilename)
    dataset = f.readlines()
    for lines in dataset:
        temp = lines.split()

        _temp = np.array([float(x) for x in temp[1:]])

        points = np.split(_temp, len(_temp) // 2)
        print(points[1])
        cv2.circle(img, before_guiyi(points[0], h, w), 5, (255, 0, 255))
        temp_rect = xywhn2xyxy(np.array([float(x) for x in temp[1:5]]), h, w)
        cv2.rectangle(img, (int(temp_rect[0]), int(temp_rect[1])), (int(temp_rect[2]), int(temp_rect[3])), (255, 0, 255),2)
    cv2.imshow(xml, img)
    cv2.waitKey(1500)
    cv2.destroyAllWindows()
