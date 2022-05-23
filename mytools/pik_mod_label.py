#!/usr/bin/env python
# coding: utf-8

import json
import os
import shutil
from tqdm import tqdm
import cv2
import glob
from matplotlib import pyplot as plt
import random

to_dir = '/media/zjx/FEAC595FAC59140D/皮带项目传承/data/模拟的洞和裂缝数据'
data_dir = "/media/zjx/FEAC595FAC59140D/皮带项目传承/data"
lb_files = sorted(glob.glob(os.path.join(data_dir, 'labels', '*', "*.txt")))
print(len(lb_files))

# ### file to find
'''
label2file = {
    0: [],
    1: [],
    2: []
}

for lf in lb_files:
    with open(lf) as f:
        lines = f.readlines()

    ll = [int(line.split()[0]) for line in lines]
    for l in ll:
        label2file[l].append(lf)

label_files = list(set(label2file[1]+label2file[2]))

for label_file in tqdm(label_files):
    src_txt = label_file
    base_txt = os.path.basename(src_txt)
    src_img = src_txt.replace('.txt', '.jpg').replace('labels', 'images')
    base_img = os.path.basename(src_img)
    det_txt = os.path.join(to_dir, base_txt)
    det_img = os.path.join(to_dir, base_img)

    shutil.copy(src_txt, det_txt)
    shutil.copy(src_img, det_img)
'''
# =================================

to_dir = '/home/yib11/VOC2028/labels_'
data_dir = "/home/yib11/VOC2028/labels"
lb_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
print(len(lb_files))

for lf in lb_files:
    with open(lf) as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line[0] == '0':
            new_lines.append('person' + line[1:])
        else:
            print(lf)
            print(line)
            # new_lines.append(line)

    src_txt = lf
    base_txt = os.path.basename(src_txt)
    # src_img = src_txt.replace('.txt', '.jpg')
    # base_img = os.path.basename(src_img)
    det_txt = os.path.join(to_dir, base_txt)
    # det_img = os.path.join(to_dir, base_img)
    with open(det_txt, 'w') as f:
        f.writelines(new_lines)
    # shutil.copy(src_img, det_img)

# ### 验证转换是否正确，随机选取训练集中的图片
'''
imgs = sorted(glob.glob(data_dir + "*.jpg"))
labels = sorted(glob.glob(data_dir + "*.txt"))

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
'''