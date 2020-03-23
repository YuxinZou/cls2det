import argparse
import os
import sys

import numpy as np
from pycocotools.coco import COCO

from detection.detector import Detector
from utils.config import Config
from utils.eval_coco import COCOeval

sys.path.append('..')




def parse_args():
    parser = argparse.ArgumentParser(description='Train RetinaNet model')
    parser.add_argument('--config', help='config file path', type=str, default='../configs/detection.py')
    parser.add_argument('--dataset', help='dataset for evaluation, train or val', type=str, default='train')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    cfg = Config.fromfile(cfg_fp)
    detect = Detector(cfg_fp)

    coco_Gt = COCO(cfg.eval[args.dataset].gt)
    if not os.path.exists(cfg.eval[args.dataset].dt):
        detect.generate_dt_json(cfg.eval[args.dataset].dt, args.dataset)
    coco_Dt = COCO(cfg.eval[args.dataset].dt)

    coco_eval = COCOeval(coco_Gt, coco_Dt, iouType='bbox')
    coco_eval.params.catIds = [cfg.voc_categories.dog]
    coco_eval.params.iouThrs = np.array([cfg.eval.iou_thres])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    main()
