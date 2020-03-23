import argparse
import sys

from detection.detector import Detector
from utils.config import Config

sys.path.append('..')



def parse_args():
    parser = argparse.ArgumentParser(description='Train RetinaNet model')
    parser.add_argument('--config', help='config file path', type=str, default='../configs/detection.py')
    parser.add_argument('--img_path', help='img for demo', type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    detect = Detector(cfg_fp)
    dets, scores, labels = detect.detect_single(args.img_path)
    print(f'{len(dets)} dog are detected')


if __name__ == '__main__':
    main()
