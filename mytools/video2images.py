import os
import glob
import cv2

img_dir = "/home/yib11/文档/datas/belt/数据源/7-现场调优数据"

video_list = [
    "/home/yib11/下载/曹科下载的文件/video_src/belt_7_p-20220311-143833.mp4",
    "/home/yib11/下载/曹科下载的文件/video_src/belt_7_p-20220311-151958.mp4"
]

i = 0
interval = 20
for video in video_list:
    cap = cv2.VideoCapture(video)
    basename = os.path.basename(video)
    basename = os.path.splitext(basename)[0]
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # if i < 1100:
            #     if i % 100 == 0:
            #         print(i)
            #     i += 1
            #     continue
            # if i > 1200:
            #     break
            if i % interval == 0:
                jpg_name = "{}_{}.jpg".format(basename, i)
                path = os.path.join(img_dir, jpg_name)
                cv2.imwrite(path, frame)
                print(i)
            i += 1
        else:
            break


