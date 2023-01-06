import os
import albumentations as A
import cv2
import matplotlib as plt
import numpy

if __name__=='__main__':
    img=cv2.imread(r"E:\Robotmaster\2021-RMUC-0417-0916\images\779.jpg")
    cv2.imshow('bbb', img)
    aug = A.Compose([
        A.OpticalDistortion(),  # 光学畸变
        # A.GridDistortion(),  # 网格畸变
        # A.CLAHE(),  # 对比度受限直方图均衡
        # A.Blur(blur_limit=15),        # # 模糊
        A.MotionBlur(blur_limit=[0,50], p=1),  # 运动模糊
        # A.MedianBlur(blur_limit=15),#中心模糊
        # A.GaussianBlur(blur_limit=15),#高斯模糊
    ])
    img = aug(image=img)['image']
    cv2.imshow('aaa',img)
    cv2.waitKey(0)
