# TODO:用于分割视频成图像
import cv2

cap = cv2.VideoCapture(r"C:\Users\59781\Downloads\Compressed\2022-西南大学GKD-大符内录\西南大学GKD-大符内录\output2.mp4")  # 读入视频文件，命名cv
n = 1  # 计数
imwrite_path=r'D:\Porject\Python_project\FourPoint\videoimage\\'
while cap.isOpened():
    ret, frame = cap.read()
    n+=1
    if n%10==0:
        cv2.imwrite(imwrite_path+str(n)+'.jpg',frame)
    cv2.imshow('frame', frame)
    cv2.waitKey(10)


cv2.destroyAllWindows()
