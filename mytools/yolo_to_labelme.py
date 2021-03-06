# coding = utf-8
"""
@ Author: kevin
@ Date: 2022/2/15 下午3:27
@ Email: caok15@sany.com.cn
"""

# import some common libraries
import glob

import numpy as np
import os

# coding here ======
# coding:utf-8

import os
import cv2
import json
import matplotlib.pyplot as plt
import base64


def parse_tta_label_459(txt_path, img_dir, save_dir):
    file_name = os.path.basename(txt_path)
    file_name = os.path.splitext(file_name)[0]
    img_path = os.path.join(img_dir, file_name + ".jpg")
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(img_path, 'rb') as f:
        image = f.read()
    image_base64 = str(base64.b64encode(image), encoding='utf-8')

    with open(txt_path, 'r') as f:
        label_info_list = f.readlines()

    version = "4.5.9"
    data_dict = dict()
    data_dict.__setitem__("version", version)
    data_dict.__setitem__("flags", {})
    data_dict["shapes"] = list()
    for label_info in label_info_list:
        label_info = label_info.strip()
        label_info = label_info.split(' ')
        class_name = label_info[0]
        c_x = int(float(label_info[1]) * w)
        c_y = int(float(label_info[2]) * h)
        b_w = int(float(label_info[3]) * w)
        b_h = int(float(label_info[4]) * h)
        x1 = int(c_x - b_w / 2)
        x2 = int(c_x + b_w / 2)
        y1 = int(c_y - b_h / 2)
        y2 = int(c_y + b_h / 2)

        points = [[x1, y1], [x2, y2]]
        shape_type = "rectangle"
        shape = {}

        shape.__setitem__("label", class_name)
        shape.__setitem__("points", points)
        shape.__setitem__("group_id", None)
        shape.__setitem__("shape_type", shape_type)
        shape.__setitem__("flags", {})
        data_dict["shapes"].append(shape)
    data_dict.__setitem__("imagePath", file_name)
    data_dict.__setitem__("imageData", image_base64)
    data_dict.__setitem__("imageHeight", h)
    data_dict.__setitem__("imageWidth", w)

    save_json_path = os.path.join(save_dir, "%s.json" % file_name)
    json.dump(data_dict, open(save_json_path, 'w'), indent=4)
    #### for view debug
    # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # plt.figure(0)
    # plt.imshow(img)
    # plt.show()


def parse_tta_label_4513(txt_path, img_dir, save_dir):
    file_name = os.path.basename(txt_path)
    file_name = os.path.splitext(file_name)[0]
    img_path = os.path.join(img_dir, file_name + ".jpg")
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(img_path, 'rb') as f:
        image = f.read()
    image_base64 = str(base64.b64encode(image), encoding='utf-8')

    with open(txt_path, 'r') as f:
        label_info_list = f.readlines()

    version = "4.5.13"
    data_dict = dict()
    data_dict.__setitem__("version", version)
    data_dict.__setitem__("imagePath", file_name)
    data_dict.__setitem__("imageData", image_base64)
    data_dict.__setitem__("imageHeight", h)
    data_dict.__setitem__("imageWidth", w)
    data_dict.__setitem__("flags", {})
    data_dict["shapes"] = list()
    for label_info in label_info_list:
        label_info = label_info.strip()
        label_info = label_info.split(' ')
        class_name = label_info[0]
        c_x = int(float(label_info[1]) * w)
        c_y = int(float(label_info[2]) * h)
        b_w = int(float(label_info[3]) * w)
        b_h = int(float(label_info[4]) * h)
        x1 = int(c_x - b_w / 2)
        x2 = int(c_x + b_w / 2)
        y1 = int(c_y - b_h / 2)
        y2 = int(c_y + b_h / 2)

        points = [[x1, y1], [x2, y2]]
        shape_type = "rectangle"
        shape = {}

        shape.__setitem__("label", class_name)
        shape.__setitem__("points", points)
        shape.__setitem__("shape_type", shape_type)
        shape.__setitem__("flags", {})
        shape.__setitem__("group_id", None)
        data_dict["shapes"].append(shape)

    save_json_path = os.path.join(save_dir, "%s.json" % file_name)
    json.dump(data_dict, open(save_json_path, 'w'), indent=4)
    #### for view debug
    # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # plt.figure(0)
    # plt.imshow(img)
    # plt.show()


def generate_labelme_prelabel(txt_dir, img_dir, save_dir):
    txt_name_list = glob.glob(os.path.join(txt_dir, "*.txt"))
    txt_name_list = [os.path.basename(it) for it in txt_name_list]
    txt_name_list.sort()
    for txt_name in txt_name_list:
        if txt_name.startswith('.'):
            continue
        print("processing -> %s" % txt_name)
        txt_path = os.path.join(txt_dir, txt_name)
        parse_tta_label_459(txt_path, img_dir, save_dir)


if __name__ == '__main__':
    img_dir = "/home/yib11/文档/datas/belt/数据源/7-现场调优数据"
    txt_dir = "/home/yib11/文档/datas/belt/数据源/7-现场调优数据"
    save_dir = "/home/yib11/文档/datas/belt/数据源/7-现场调优数据"
    generate_labelme_prelabel(txt_dir, img_dir, save_dir)
