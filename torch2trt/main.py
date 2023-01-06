import os
import sys
import time

import cv2
import copy
import torch
import argparse

root_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))  # 项目根路径：获取当前路径，再上级路径
sys.path.append(root_path)  # 将项目根路径写入系统路径
from utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh
from utils.datasets import letterbox
from detect_face import scale_coords_landmarks, show_results
from torch2trt.trt_model import TrtModel

cur_path = os.path.abspath(os.path.dirname(__file__))


def img_process(img_path, long_side=640, stride_max=32):
    '''
    图像预处理
    '''
    orgimg = cv2.imread(img_path)
    img0 = copy.deepcopy(orgimg)
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = long_side / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(long_side, s=stride_max)  # check img_size

    img = letterbox(img0, new_shape=imgsz, auto=False)[0]  # auto True最小矩形   False固定尺度
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, orgimg


def video_process(img, long_side=640, stride_max=32):
    '''
    图像预处理
    '''
    orgimg = copy.deepcopy(img)
    img0 = copy.deepcopy(orgimg)
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = long_side / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(long_side, s=stride_max)  # check img_size

    img = letterbox(img0, new_shape=imgsz, auto=False)[0]  # auto True最小矩形   False固定尺度
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, orgimg


def img_vis(img, orgimg, pred, vis_thres=0.6):
    '''
    预测可视化
    vis_thres: 可视化阈值
    '''
    loop_start = time.time()

    print('img.shape: ', img.shape)
    print('orgimg.shape: ', orgimg.shape)

    no_vis_nums = 0
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            temp=det[:, :4]
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

            for j in range(det.size()[0]):

                if det[j, 4].cpu().numpy() < vis_thres:
                    no_vis_nums += 1
                    continue

                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:15].view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                orgimg = show_results(orgimg, xyxy, conf, landmarks, class_num)

    loop_end = time.time()
    total_time = loop_end -loop_start  # 使用getTickFrequency()更加准确
    running_FPS = int(1 / total_time)  # 帧率取整
    cv2.putText(orgimg, str(running_FPS), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 5)
    cv2.imshow('result', orgimg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str,
                        default=r"/media/yuesang/KESU/Robotmaster/dataset/data/images/train/0dd119c24d073325871696e372069ea6.jpg",
                        help='img path')
    parser.add_argument('--trt_path', type=str, default=r"../runs/train/rm/weights/best2.trt", help='trt_path')
    parser.add_argument('--output_shape', type=list, default=[1, 25200, 19],
                        help='input[1,3,640,640] ->  output[1,25200,16]')
    parser.add_argument('--video', default=r"/media/yuesang/KESU/Robotmaster/dataset/video/video/2.mp4",
                        help='using video')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='')
    opt = parser.parse_args()
    model = TrtModel(opt.trt_path)
    if opt.video!="":
        cap = cv2.VideoCapture(opt.video)
        while cap.isOpened():

            ret, frame = cap.read()
            video_frame, video_origin = video_process(frame)

            pred = model(video_frame.numpy()).reshape(opt.output_shape)  # forward

            # Apply NMS
            pred = non_max_suppression_face(torch.from_numpy(pred), opt.conf_thres, opt.iou_thres)
            img_vis(video_frame, video_origin, pred)
            cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()
    if opt.img_path != "":
        img, orgimg = img_process(opt.img_path)
        pred = model(img.numpy()).reshape(opt.output_shape)  # forward
        model.destroy()

        # Apply NMS
        pred = non_max_suppression_face(torch.from_numpy(pred), opt.conf_thres, opt.iou_thres)
        # ============可视化================
        img_vis(img, orgimg, pred)
        cv2.waitKey(3000)