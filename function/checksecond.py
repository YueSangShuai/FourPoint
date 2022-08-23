# TODO:用于判断生成的数据是否合法主要是用于把数据集中的点反归一化后重新展现在图片上
import os
import cv2
import numpy as np
import torch

def before_guiyi(x, w, h):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = int(w * x[0])
    y[1] = int(h * x[1])
    y = y.astype(np.int64)
    return y


image_path = r'E:\Robotmaster\dataset\mydata\images\train\\'
xml_path= r'E:\Robotmaster\kaiyuanzhuagnjiaban\2021-RMUC-SYHK-1021\labels\\'  # 数据原图路径


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

h=1920
w=1080

for xml in xml_filename:
    xmlfilename=xml_path+"%s.txt"%xml
    imgfilename = image_path+"%s.png" % xml
    img=cv2.imread(imgfilename)
    f=open(xmlfilename)
    dataset = f.readlines()
    for lines in dataset:
        temp = lines.split()

        _temp = np.array([float(x) for x in temp[5:]])

        points = np.split(_temp, len(_temp) // 2)

        print(points)
        cv2.circle(img, before_guiyi(points[1], h, w), 5, (255, 0, 255))

    cv2.imshow(xml, img)
    cv2.waitKey(1500)
    cv2.destroyAllWindows ()

