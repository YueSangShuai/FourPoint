# TODO：用于删除xml存在但是图片不存在的图片


import os
from pathlib import Path

from tqdm import tqdm

xml_path = r'/media/yuesang/G/Robotmaster/shangjiao/labels/'  # 标注文件路径
image_path = r'/media/yuesang/G/Robotmaster/shangjiao/images/'  # 数据原图路径

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

for difference in set(xml_filename)^set(img_filename):
    path=image_path+difference+'.jpg'
    os.remove(path)

