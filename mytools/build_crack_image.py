#!/usr/bin/env python
# coding: utf-8


import cv2
import os
import os.path as osp
import json
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import tqdm
import copy


class BeltRoi:
    def __init__(self, path, label, box, width, height, src_img):
        self.img_path = path
        self.label = label
        self.box = box
        self.width = width
        self.height = height
        self.src_img = src_img


class CrackRoi:
    def __init__(self, path, label, contour, width, height, src_img, roi_mask):
        self.img_path = path
        self.label = label
        self.contour = contour
        self.width = width
        self.height = height
        self.src_img = src_img
        self.roi_mask = roi_mask

    def flip_horizontally(self):
        self.src_img = cv2.flip(self.src_img, 1)
        self.roi_mask = cv2.flip(self.roi_mask, 1)

    def flip_vertically(self):
        self.src_img = cv2.flip(self.src_img, 0)
        self.roi_mask = cv2.flip(self.roi_mask, 0)

    def rotate_90(self):
        self.src_img = cv2.rotate(self.src_img, cv2.ROTATE_90_CLOCKWISE)
        self.roi_mask = cv2.rotate(self.roi_mask, cv2.ROTATE_90_CLOCKWISE)
        self.width, self.height = self.height, self.width

    def scale(self, r):
        self.src_img = cv2.resize(self.src_img, (0, 0), fx=r, fy=r)
        self.roi_mask = cv2.resize(self.roi_mask, (0, 0), fx=r, fy=r)
        self.width, self.height = self.width * r, self.height * r

    def motion_blur(self, degree=12, angle=90):
        """
        运动模糊 (degree越大，模糊程度越高)
        """
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M,
                                            (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(self.src_img, -1, motion_blur_kernel)
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        self.src_img = np.array(blurred, dtype=np.uint8)


def get_belt_rois(roi_img_dir):
    """
    获取皮带roi区域
    """
    img_files = sorted(glob.glob(roi_img_dir + "*.jpg"))
    annots = sorted(glob.glob(roi_img_dir + "*.json"))
    rois = []
    for img_file, annot_file in zip(img_files, annots):
        with open(annot_file) as f:
            json_data = json.load(f)
        for shape in json_data["shapes"]:
            points = shape["points"]
            x1 = int(round(points[0][0]))
            y1 = int(round(points[0][1]))
            x2 = int(round(points[1][0]))
            y2 = int(round(points[1][1]))
            (x1, x2) = (x2, x1) if x2 < x1 else (x1, x2)
            (y1, y2) = (y2, y1) if y2 < y1 else (y1, y2)
            w, h = x2 - x1, y2 - y1
            box = [x1, y1, x2, y2]
            label = shape["label"]
            img = cv2.imread(img_file)
            rois.append(BeltRoi(img_file, label, box, w, h, img))
    return rois


def get_crack_rois(roi_img_dir):
    """
    获取破损roi区域
    """
    img_files = sorted(glob.glob(roi_img_dir + "*.jpg"))
    annots = sorted(glob.glob(roi_img_dir + "*.json"))
    rois = []
    for img_file, annot_file in zip(img_files, annots):
        with open(annot_file) as f:
            json_data = json.load(f)
        for shape in json_data["shapes"]:
            points = np.array(shape["points"], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(points)
            label = shape["label"]
            img = cv2.imread(img_file)
            msk = np.zeros(img.shape, dtype=np.uint8)
            cv2.fillPoly(msk, [points], (255, 255, 255))
            rois.append(CrackRoi(img_file, label, points, w, h, img, msk))
    return rois


def merge(c_roi, b_roi):
    """
    融合
    """
    # 获取皮带融合区域
    x1, y1, x2, y2 = b_roi.box
    a_w = b_roi.width
    a_h = b_roi.height

    # c_roi 初始信息
    # cv2.imshow('img_roi', c_roi.src_img)
    # cv2.imshow('mask', c_roi.roi_mask)
    # cv2.waitKey()

    # data augmentation
    # 左右翻转
    c_roi.flip_horizontally()
    # cv2.imshow('img_roi', c_roi.src_img)
    # cv2.imshow('mask', c_roi.roi_mask)
    # cv2.waitKey()
    # 上下翻转
    c_roi.flip_vertically()
    # cv2.imshow('img_roi', c_roi.src_img)
    # cv2.imshow('mask', c_roi.roi_mask)
    # cv2.waitKey()
    # 顺时针90度
    # c_roi.rotate_90()
    # cv2.imshow('img_roi', c_roi.src_img)
    # cv2.imshow('mask', c_roi.roi_mask)
    # cv2.waitKey()
    # 缩放
    scale = random.uniform(1.1, 2.1)
    print(scale)
    c_roi.scale(scale)
    if c_roi.width > a_w:
        print('width too big')
        scale_w = random.randint(int(a_w*0.2), int(a_w*0.8))
        scale = scale_w / c_roi.width
        print(scale)
        c_roi.scale(scale)
    if c_roi.height > a_h:
        print('height too big')
        scale_h = random.randint(int(a_h*0.2), int(a_h*0.8))
        scale = scale_h / c_roi.height
        print(scale)
        c_roi.scale(scale)
    if c_roi.width < 50 and c_roi.height < 50:
        print('width and height too small')
        scale = random.uniform(1.1, 1.6)
        print(scale)
        c_roi.scale(scale)

    # cv2.imshow('img_roi', c_roi.src_img)
    # cv2.imshow('mask', c_roi.roi_mask)
    # cv2.waitKey()
    # 模糊
    c_roi.motion_blur()

    try:
        c_l, c_r = x1 + c_roi.width // 2, x2 - c_roi.width // 2
        c_t, c_b = y1 + c_roi.height // 2, y2 - c_roi.height // 2

        if c_l > c_r:
            print('c_l, c_r', c_l, c_r)
            print('x1, x2, a_w', x1, x2, a_w)
            print('c_roi.width // 2, c_roi.width // 2', c_roi.width // 2, c_roi.width // 2)
        if c_t > c_b:
            print('c_t, c_b', c_t, c_b)
            print('y1, y2, a_h', y1, y2, a_h)
            print('c_roi.height // 2, c_roi.height // 2', c_roi.height // 2, c_roi.height // 2)
        cx = random.randint(c_l, c_r)
        cy = random.randint(c_t, c_b)

        # cv2.imshow('b_roi.src_img', b_roi.src_img)
        # cv2.waitKey()

        # kevin 泊松融合
        out_res = cv2.seamlessClone(c_roi.src_img, b_roi.src_img, c_roi.roi_mask, (cx, cy), cv2.NORMAL_CLONE)

        # cv2.imshow('out_res', out_res)
        # cv2.waitKey()

    except Exception as e:
        print(e)
        return

    x1 = int(cx - c_roi.width // 2)
    y1 = int(cy - c_roi.height // 2)
    x2 = int(cx + c_roi.width // 2)
    y2 = int(cy + c_roi.height // 2)
    out_mark = out_res.copy()
    out_mark = cv2.rectangle(out_mark, (x1, y1), (x2, y2), (0, 0, 255), 2)

    h, w = b_roi.src_img.shape[:2]
    out_coords = [round(cx / w, 6), round(cy / h, 6), round(c_roi.width / w, 6), round(c_roi.height / h, 6)]
    out_coords = [str(x) for x in out_coords]
    return out_res, out_mark, out_coords


if __name__ == '__main__':

    crack_img_dir = "/home/yib11/文档/datas/belt/数据源/rois/"  # 原始破损样本目录
    crack_rois = get_crack_rois(crack_img_dir)

    belt_img_dir = "/home/yib11/文档/datas/belt/数据源/belt_bg/"
    belt_rois = get_belt_rois(belt_img_dir)
    belt_rois = belt_rois[:1]

    # 创建结果存放目录
    out_dir = os.path.join(belt_img_dir, "res")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for i, belt_roi in enumerate(belt_rois):
        print(belt_roi.img_path)
        crack_rois_sample = random.sample(crack_rois, 8)
        crack_rois_sample = copy.copy(crack_rois)
        for j, crack_roi in enumerate(crack_rois_sample):
            print(crack_roi.img_path)
            res = merge(crack_roi, belt_roi)
            if res is None:
                continue
            merged, mark, coords = res

            # kevin
            # show_fx = show_fy = 0.5
            # merged4show = cv2.resize(merged, (0, 0), fx=show_fx, fy=show_fy)
            # mark4show = cv2.resize(mark, (0, 0), fx=show_fx, fy=show_fy)
            # cv2.imshow('merged', merged4show)
            # cv2.imshow('mark', mark4show)
            # cv2.waitKey()

            # 融合结果图片
            img_name = "{}_{}.jpg".format(i, j)
            img_path = os.path.join(out_dir, img_name)
            cv2.imwrite(img_path, merged)
            # 用于融合图片正确性检查
            mark_name = "{}_{}m.jpg".format(i, j)
            mark_path = os.path.join(out_dir, mark_name)
            cv2.imwrite(mark_path, mark)
            # 生成破损区域标注
            txt_name = "{}_{}.txt".format(i, j)
            txt_path = os.path.join(out_dir, txt_name)
            with open(txt_path, "w") as f:
                # label_id = label_map[roi.label]
                line = " ".join([crack_roi.label] + coords) + "\n"
                f.write(line)
