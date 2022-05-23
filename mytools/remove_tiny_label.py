import os
from tqdm import tqdm
import shutil
import imghdr
from mimetypes import guess_type
import glob
import hashlib
import imghdr
import cv2
import numpy as np


def get_filelist(root_path, ext):
    original_images = []
    for root, dirs, filenames in os.walk(root_path):
        if dirs:
            for dir_ in dirs:
                # print('dir', dir_)
                for file in glob.glob(os.path.join(root, dir_, '*.' + ext)):
                    original_images.append(file)
        else:
            for file in glob.glob(os.path.join(root, '*.' + ext)):
                original_images.append(file)

    original_images = sorted(original_images)
    print('tatol num:', len(original_images))

    return original_images


def chk_gt_size(root_path, file_end):
    print('chk_size_wh function get in')
    original_images = get_filelist(root_path, file_end)

    small_gt_images = 0
    tar_label = '1'
    w_th = 0.05
    h_th = 0.05
    for filename in tqdm(original_images):
        src_img = filename
        src_txt = src_img.replace(file_end, 'txt')
        src_json = src_img.replace(file_end, 'json')
        # print(b_size)
        with open(src_txt) as f:
            lines = f.readlines()

        lines = [line for line in lines if line[0] == tar_label]

        for line in lines:
            _, _, _, w_gt, h_gt = line.split()
            w_gt, h_gt = float(w_gt), float(h_gt)
            if h_gt < h_th and w_gt < w_th:
                print('too small gt! remove: ', src_img)
                print('gt w: {:.3f}  gt h:{:.3f}'.format(w_gt, h_gt))
                os.remove(src_txt)
                os.remove(src_img)
                os.remove(src_json)
                small_gt_images += 1
                continue
            if (1.77*w_gt < h_gt) and (h_gt * w_gt < 0.8 * h_th * w_th):   # | type
                print('too small gt aero! remove: ', src_img)
                print('gt aero: {:.5f}'.format(w_gt*h_gt))
                os.remove(src_txt)
                os.remove(src_img)
                os.remove(src_json)
                small_gt_images += 1
            if (1.77*w_gt >= h_gt) and (h_gt * w_gt < 0.6 * h_th * w_th):     # - type
                print('too small gt aero! remove: ', src_img)
                print('gt aero: {:.5f}'.format(w_gt*h_gt))
                os.remove(src_txt)
                os.remove(src_img)
                os.remove(src_json)
                small_gt_images += 1
        else:
            pass
    print('too small images:', small_gt_images)


if __name__ == '__main__':
    root_path = '/media/zjx/FEAC595FAC59140D/皮带项目传承/data/数据源/模拟的洞和裂缝数据合并1类'
    file_ext = 'jpg'

    chk_gt_size(root_path, file_ext)
