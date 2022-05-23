import argparse
import glob
import numpy as np
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (check_img_size, cv2, increment_path, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync


class Config:
    def __init__(self):
        self.weights = './runs/train/sany_final.pt'
        self.source = './data/images_c'
        self.data = './data/sany.yaml'
        self.imgsz = [640, 640]
        self.device = '0'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.line_thickness = 3  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False
        self.dnn = False


def plot_res_on_frame(frame, res):
    names = ['hole', 'hopper']
    hide_labels = False
    hide_conf = False

    annotator = Annotator(frame, line_width=3, example=str(names))
    if res:
        # 取出 模型检测结果 异常状态码
        det, stat_code_list = res

        # 绘制模型检测结果
        for *xyxy, conf, cls in reversed(det):
            # Add bbox to image
            c = int(cls)  # integer class
            label = None if hide_labels else (
                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))
        # Stream results
        frame = annotator.result()

    return frame


class Detector(object):
    def __init__(self, cfg):
        self.model = None
        self.device = None
        self.imgsz = None
        self.stride = None
        self.names = None
        self.pt = None
        self.cfg = cfg
        self.count = 0

    def init_model(self):
        """
        initialize model.

        Returns:
            model: initialized model
        """
        # Load model
        self.device = select_device(self.cfg.device)
        model = DetectMultiBackend(weights=self.cfg.weights, device=self.device, dnn=self.cfg.dnn,
                                   data=self.cfg.data, fp16=self.cfg.half)
        self.stride, self.names, self.pt = model.stride, model.names, model.pt
        self.model = model
        self.imgsz = check_img_size(self.cfg.imgsz, s=self.stride)  # check image size
        bs = 1  # batch_size
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup


    @torch.no_grad()
    def inference(self, im0s):

        # Padded resize
        img = letterbox(im0s, self.imgsz, stride=self.stride, auto=self.pt)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # Run inference
        t1 = time_sync()
        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()

        # Inference
        pred = self.model(im, augment=self.cfg.augment, visualize=False)
        t3 = time_sync()

        # NMS
        pred = non_max_suppression(pred, self.cfg.conf_thres,
                                   self.cfg.iou_thres,
                                   classes=None,
                                   agnostic=self.cfg.agnostic_nms,
                                   max_det=self.cfg.max_det)

        det = pred[0]  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
        else:
            det = None

        return det

    def run_on_img_dir(self,
                       project=ROOT / 'runs/detect',  # save results to project/name
                       name='exp',  # save results to project/name
                       exist_ok=False,  # existing project/name ok, do not increment
                       ):

        self.init_model()

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # Dataloader
        files = sorted(glob.glob(os.path.join(self.cfg.source, '*')))  # glob

        # ====
        for path in files:
            # Read image
            self.count += 1
            im0s = cv2.imread(path)  # BGR
            assert im0s is not None, f'Image Not Found {path}'

            det = self.inference(im0s)

            # Process predictions

            p, im0 = path, im0s.copy()

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            annotator = Annotator(im0, line_width=self.cfg.line_thickness, example=str(self.names))
            if len(det):

                # Stream results
                im0 = plot_res_on_frame(im0s, [det, None])

                # Save results (image with detections)
                cv2.imshow('img', im0)
                cv2.waitKey()


if __name__ == "__main__":
    opt = Config()
    d = Detector(opt)
    d.run_on_img_dir()
