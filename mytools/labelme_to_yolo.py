#!/usr/bin/env python
# coding: utf-8

import json
import os
import cv2
import glob
from matplotlib import pyplot as plt
import random

data_dir = "/home/yib11/文档/datas/belt/数据源/7-现场调优数据"
js_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
print(js_files)

# ### 转换：labelme to yolo

label_map = {
    "roller": "roller",
    "crack": "crack",
    "person": "person",
    "hat": "hat",
    "0": "roller",
    "1": "crack",
    "2": "person",
    "3": "hat"
}

for json_file in js_files:
    with open(json_file) as f:
        json_data = json.load(f)

    h_img = json_data["imageHeight"]
    w_img = json_data["imageWidth"]

    lines = []
    for shape in json_data["shapes"]:
        label = shape["label"]
        points = shape["points"]
        x1 = points[0][0]
        y1 = points[0][1]
        x2 = points[1][0]
        y2 = points[1][1]

        (left, right) = (x1, x2) if x1 < x2 else (x2, x1)
        (top, bottom) = (y1, y2) if y1 < y2 else (y2, y1)
        assert (left < right) and (top < bottom)

        w = right - left
        h = bottom - top
        cx = left + w / 2
        cy = top + h / 2

        coord = [round(cx / w_img, 6), round(cy / h_img, 6), round(w / w_img, 6), round(h / h_img, 6)]
        line = [label_map[label]] + coord
        line = [str(x) for x in line] + ["\n"]
        lines.append(" ".join(line))

    basename = os.path.basename(json_file)
    label_file = basename.replace(".json", ".txt")
    save_path = os.path.join(data_dir, label_file)
    # with open(save_path, "a+") as f:
    with open(save_path, "w") as f:
        f.writelines(lines)

# ### 验证转换是否正确，随机选取训练集中的图片

imgs = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
labels = sorted(glob.glob(os.path.join(data_dir, "*.txt")))

idx = random.randint(0, len(imgs))
img_file, label_file = imgs[idx], labels[idx]

img = cv2.imread(img_file)
h_img, w_img = img.shape[:2]

with open(label_file) as f:
    lines = f.readlines()
    lines = [line.strip().split(" ") for line in lines]

for line in lines:
    coord = list(map(float, line[1:]))
    cx = int(coord[0] * w_img)
    cy = int(coord[1] * h_img)
    w = int(coord[2] * w_img)
    h = int(coord[3] * h_img)
    cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (0, 0, 255), 2)

plt.figure(figsize=(12, 8))
plt.imshow(img[:, :, ::-1])
plt.show()
