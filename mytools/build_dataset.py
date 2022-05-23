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

dst_root_dir = '/home/yib11/文档/datas/belt/data'
src_root_dir = "/home/yib11/文档/datas/belt/数据源"
subset_list = ['1-托辊', '2-模拟破损照片粘贴数据', '3-模拟的洞和裂缝数据合并1类',
               '4-安全帽和人', '5-模拟破损照片粘贴数据2', '6-模拟的裂缝数据', '7-现场调优数据']
prefix_list = ['tk_', 'zp_', 'mn_', 'yw_', 'zp2_', 'mn2_', 'ty_']
train_per = 0.8

dir_path = os.path.join(dst_root_dir, 'images')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
dir_path = os.path.join(dst_root_dir, 'images', 'train')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
dir_path = os.path.join(dst_root_dir, 'images', 'val')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
dir_path = os.path.join(dst_root_dir, 'labels')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
dir_path = os.path.join(dst_root_dir, 'labels', 'train')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
dir_path = os.path.join(dst_root_dir, 'labels', 'val')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

for dir_, prefix in zip(subset_list, prefix_list):
    src_txt_list = glob.glob(os.path.join(src_root_dir, dir_, "*.txt"))
    total_num = len(src_txt_list)
    print(dir_, total_num)
    random.shuffle(src_txt_list)
    train_point = int(total_num * train_per)
    print('train:', train_point, 'val:', total_num-train_point)
    for idx, src_txt in enumerate(src_txt_list):
        base_txt = prefix + os.path.basename(src_txt)

        src_img = src_txt.replace('.txt', '.jpeg')
        base_img = prefix + os.path.basename(src_img)
        if idx < train_point:
            det_txt = os.path.join(dst_root_dir, 'labels', 'train', base_txt)
            det_img = os.path.join(dst_root_dir, 'images', 'train', base_img)
        else:
            det_txt = os.path.join(dst_root_dir, 'labels', 'val', base_txt)
            det_img = os.path.join(dst_root_dir, 'images', 'val', base_img)
        if os.path.exists(det_txt):
            pass
            print(det_txt)
        shutil.copy(src_txt, det_txt)
        shutil.copy(src_img, det_img)

print('=======================')
train_txt_list = glob.glob(os.path.join(dst_root_dir, 'labels', 'train', '*.txt'))
print('train_txt_list ', len(train_txt_list))
val_txt_list = glob.glob(os.path.join(dst_root_dir, 'labels', 'val', '*.txt'))
print('val_txt_list ', len(val_txt_list))
