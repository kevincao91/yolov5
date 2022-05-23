import os

from PIL import Image
from tqdm import tqdm
import shutil
import imghdr
from mimetypes import guess_type
import glob
import hashlib
import imghdr
import cv2
import numpy as np


def get_filelist(root_dir, suffix):  # 查找根目录，文件后缀
    res = []
    for root, directory, files in os.walk(root_dir):  # =>当前根,根下目录,目录下的文件
        for filename in files:
            name, suf = os.path.splitext(filename)  # =>文件名,文件后缀
            if suf == suffix:
                res.append(os.path.join(root, filename))  # =>吧一串字符串组合成路径

    res = sorted(res)
    print('total num:', len(res))

    return res


def remove_same_piture_by_get_md5(root_path, file_end):
    original_images = get_filelist(root_path, file_end)

    same_img = 0
    md5_list = []
    for filename in tqdm(original_images):
        m = hashlib.md5()
        mfile = open(filename, "rb")
        m.update(mfile.read())
        mfile.close()
        md5_value = m.hexdigest()
        # print(md5_value)
        if (md5_value in md5_list):
            os.remove(filename)
            same_img += 1
        else:
            md5_list.append(md5_value)
    print('remove same images by md5:', same_img)


def remove_simillar_picture_by_perception_hash(path, file_end):
    original_images = get_filelist(root_path, file_end)

    hash_dic = {}
    hash_list = []
    simillar_imgs = 0
    for img_name in tqdm(original_images):
        try:
            img = cv2.imread(os.path.join(path, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            continue

        img = cv2.resize(img, (8, 8))

        avg_np = np.mean(img)
        img = np.where(img > avg_np, 1, 0)
        hash_dic[img_name] = img
        if len(hash_list) < 1:
            hash_list.append(img)
        else:
            for i in hash_list:
                flag = True
                dis = np.bitwise_xor(i, img)

                if np.sum(dis) < 5:
                    flag = False
                    print(os.path.join(path, img_name))
                    # os.remove(os.path.join(path, img_name))
                    simillar_imgs += 1
                    break
            if flag:
                hash_list.append(img)
    print('remove simillar imgs num:', simillar_imgs)


def is_valid_jpg(jpg_file):
    """
    判断JPG文件下载是否完整
    """
    if jpg_file.split('.')[-1].lower() == 'jpeg':
        with open(jpg_file, 'rb') as f:
            f.seek(-2, 2)
            read_byte = f.read()
            # print(read_byte)
            if read_byte == b'\xff\xd9':
                return True
            else:
                False
    else:
        return True


def chk_valid_jpg(root_path, file_end):
    print('chk_valid_jpg function get in')
    original_images = get_filelist(root_path, file_end)

    not_valid_jpg = 0
    for filename in tqdm(original_images):
        if not is_valid_jpg(filename):
            # print(filename)
            os.remove(filename)
            not_valid_jpg += 1
    print('remove not valid jpg num:', not_valid_jpg)


def chk_error_type(root_path, file_end):
    print('chk_error_type function get in')
    original_images = get_filelist(root_path, file_end)

    # path 1 ====================
    '''
    error_images = 0
    for filename in tqdm(original_images):
        # print(filename)
        file_end = os.path.split(filename)[-1].lower().split('.')[-1]
        # print(file_end)
        check = imghdr.what(filename)
        # print(check)
        if check is None:
            # print('error file! remove', filename)
            error_images += 1
            # srcfile = filename
            # dstfile = filename.replace('.'+file_end, 'error.'+file_end)
            # shutil.move(srcfile,dstfile)          #移动文件
            os.remove(filename)
        elif check != file_end:
            srcfile = filename
            dstfile = filename.replace(file_end, check)
            # print('{} rename to {}'.format(srcfile, dstfile))
            shutil.move(srcfile, dstfile)  # 移动文件
        else:
            pass
    '''

    # path 2 ====================
    error_images = 0
    for filename in tqdm(original_images):
        # print(filename)
        mimetype, _ = guess_type(filename)
        maintype, subtype = mimetype.split('/')
        # print(maintype, subtype)
        if maintype != 'image':
            # print('error file! remove', filename)
            error_images += 1
        elif subtype != file_end[1:]:
            srcfile = filename
            dstfile = filename.replace(file_end, '.'+subtype)
            # print('{} rename to {}'.format(srcfile, dstfile))
            shutil.move(srcfile, dstfile)  # 移动文件
        else:
            pass

    print('error images num:', error_images)


# 获取文件大小
def getFileSize(path):
    try:
        size = os.path.getsize(path)
        return size
    except Exception as err:
        print(err)


def chk_size_wh(root_path, file_end):
    print('chk_size_wh function get in')
    original_images = get_filelist(root_path, file_end)

    small_images = 0
    size_th = 1024 * 100  # 100Kb
    w_th = 224
    h_th = 224
    for filename in tqdm(original_images):
        b_size = getFileSize(filename)
        # print(b_size)
        if b_size < size_th:
            image = cv2.imread(filename)
            if image is None:
                os.remove(filename)  # gif opencv 不能处理
                small_images += 1
                continue
            height = image.shape[0]
            width = image.shape[1]
            if height < h_th or width < w_th:
                # print('too small file! remove', filename)
                # file_end = os.path.split(filename)[-1].lower().split('.')[-1]
                # srcfile = filename
                # dstfile = filename.replace('.'+file_end, 'small.'+file_end)
                # shutil.move(srcfile,dstfile)          #移动文件
                os.remove(filename)
                small_images += 1
        else:
            pass
    print('too small images:', small_images)


def rewrite_img(root_path, file_end):
    print('chk_error_type function get in')
    original_images = get_filelist(root_path, file_end)

    error_images = 0
    for filename in tqdm(original_images):
        print(filename)
        srcfile = filename
        dstfile = filename.replace('4-安全帽和人', '4-安全帽和人_')
        print('{} rename to {}'.format(srcfile, dstfile))
        image = cv2.imread(srcfile)
        cv2.imwrite(dstfile, image)
        src_size = getFileSize(srcfile)
        dst_size = getFileSize(dstfile)
        print(dst_size-src_size)

    print('error images num:', error_images)


if __name__ == '__main__':
    root_path = "/home/yib11/文档/datas/belt/数据源/7-现场调优数据"
    img_ext = '.jpg'

    chk_error_type(root_path, img_ext)
    # chk_valid_jpg(root_path, img_ext)
    # chk_size_wh(root_path, img_ext)
    # remove_same_piture_by_get_md5(root_path, img_ext)
    # remove_simillar_picture_by_perception_hash(root_path, img_ext)

    # solution warning: Corrupt JPEG data: 2 extraneous bytes before marker 0xd9
    # rewrite_img(root_path, img_ext)
