# TODO:用于判断数据集中总共有多少类别
import os
from collections import Counter


def count(label_path):
    files = os.listdir(label_path)
    classes = []
    for file in files:
        f = open(label_path + file)
        dataset = f.readlines()
        for line in dataset:
            tempdata1 = line.split()
            classes.append(tempdata1[0])
    temp = set(classes)
    classes = list(temp)
    dict = []
    for i in range(len(classes)):
        tempdict = [int(classes[i]), i]
        dict.append(tempdict)

    return dict


if __name__ == '__main__':
    label_path = r'C:\Users\59781\Desktop\mydata\mydata\train\labels\\'
    print(count(label_path))
    print(len(count(label_path)))
