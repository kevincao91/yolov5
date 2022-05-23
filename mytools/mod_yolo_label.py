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


def mod_dataset():
    src_dir = "/home/yib11/文档/datas/belt/data/labels/train"
    dst_dir = '/home/yib11/文档/datas/belt/data/labels/train'

    lb_files = sorted(glob.glob(os.path.join(src_dir, "*.txt")))
    print(len(lb_files))

    label_map = {
        "roller": "0",
        "crack" : "1",
        "person": "2",
        "hat"   : "3"
    }

    for lf in lb_files:
        with open(lf) as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            fields = line.split()
            label = fields[0]
            if label in label_map:
                fields[0] = label_map[label]
                new_lines.append(' '.join(fields)+'\n')
            else:
                print(lf)
                print(line)

        src_txt = lf
        base_txt = os.path.basename(src_txt)
        det_txt = os.path.join(dst_dir, base_txt)

        new_lines.sort()
        with open(det_txt, 'w') as f:
            f.writelines(new_lines)

    src_dir = "/home/yib11/文档/datas/belt/data/labels/val"
    dst_dir = '/home/yib11/文档/datas/belt/data/labels/val'

    lb_files = sorted(glob.glob(os.path.join(src_dir, "*.txt")))
    print(len(lb_files))

    label_map = {
        "roller": "0",
        "crack": "1",
        "person": "2",
        "hat": "3"
    }

    for lf in lb_files:
        with open(lf) as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            fields = line.split()
            label = fields[0]
            if label in label_map:
                fields[0] = label_map[label]
                new_lines.append(' '.join(fields) + '\n')
            else:
                print(lf)
                print(line)

        src_txt = lf
        base_txt = os.path.basename(src_txt)
        det_txt = os.path.join(dst_dir, base_txt)

        new_lines.sort()
        with open(det_txt, 'w') as f:
            f.writelines(new_lines)


def mod_dir():
    src_dir = "/home/yib11/文档/datas/belt/数据源/7-现场调优数据"
    dst_dir = '/home/yib11/文档/datas/belt/数据源/7-现场调优数据'

    lb_files = sorted(glob.glob(os.path.join(src_dir, "*.txt")))
    print(len(lb_files))

    label_map = {
        "0": "roller",
        "1": "crack",
        "2": "person",
        "3": "hat"
    }

    for lf in lb_files:
        with open(lf) as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            fields = line.split()
            label = fields[0]
            if label in label_map:
                fields[0] = label_map[label]
                new_lines.append(' '.join(fields) + '\n')
            else:
                print(lf)
                print(line)
                # new_lines.append(line)

        src_txt = lf
        base_txt = os.path.basename(src_txt)
        det_txt = os.path.join(dst_dir, base_txt)

        new_lines.sort()
        with open(det_txt, 'w') as f:
            f.writelines(new_lines)


if __name__ == '__main__':
    mod_dataset()
    # mod_dir()
