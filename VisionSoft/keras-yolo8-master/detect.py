# YOLOv8 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv8 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov8s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov8s.pt                 # PyTorch
                                 yolov8s.torchscript        # TorchScript
                                 yolov8s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov8s_openvino_model     # OpenVINO
                                 yolov8s.engine             # TensorRT
                                 yolov8s.mlmodel            # CoreML (macOS-only)
                                 yolov8s_saved_model        # TensorFlow SavedModel
                                 yolov8s.pb                 # TensorFlow GraphDef
                                 yolov8s.tflite             # TensorFlow Lite
                                 yolov8s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov8s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from ultralytics import YOLO


def run(
        weights=ROOT / 'yolov8n.pt',  # model path or triton URL
        source="0",  # ROOT / '0',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf=0.25,  # confidence threshold
        iou=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        # classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        # update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,  # High resolution masks
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # Load model
    device = select_device(device)
    model = YOLO(weights)

    model.predict(source=source, save=not nosave,iou=iou, conf=conf, save_txt=save_txt, save_conf=save_conf,
                  save_crop=save_crop, agnostic_nms=agnostic_nms, augment=augment,
                  visualize=visualize, project=project, name=name, exist_ok=exist_ok,
                  line_thickness=line_thickness, hide_labels=hide_labels, hide_conf=hide_conf, half=half, dnn=dnn,
                  vid_stride=vid_stride, retina_masks=retina_masks)

def parse_opt():
    '''
    - `weights`: 模型权重文件的路径或 Triton URL。
    - `source`: 图像或视频文件的路径或 URL，或者摄像头编号（0 表示默认摄像头）。
    - `data`: 数据集配置文件的路径，其中包括类别名称。
    - `imgsz`: 图像尺寸（高度，宽度），用于推断过程中的图像缩放。
    - `conf`: 置信度阈值。
    - `iou`: 非极大值抑制 (NMS) 的重叠阈值。
    - `max_det`: 每张图像中最多的检测框数量。
    - `device`: 使用的设备，可以是 cuda 设备（如 0 表示使用第一张显卡）或 cpu。
    - `view_img`: 是否显示检测结果图像。
    - `save_txt`: 是否将检测结果保存为 .txt 文件。
    - `save_conf`: 是否保存检测结果中的置信度。
    - `save_crop`: 是否保存检测结果中的裁剪框。
    - `nosave`: 是否不保存图像或视频。
    - `agnostic_nms`: 是否进行类别无关的 NMS。
    - `augment`: 是否进行数据增强。
    - `visualize`: 是否在推断过程中可视化特征。
    - `project`: 结果保存的项目路径。
    - `name`: 结果保存的项目名称。
    - `exist_ok`: 如果项目路径已经存在，是否覆盖它。
    - `line_thickness`: 绘制边界框的线条粗细。
    - `hide_labels`: 是否隐藏标签。
    - `hide_conf`: 是否隐藏置信度。
    - `half`: 是否使用 FP16 的半精度推断。
    - `dnn`: 是否使用 OpenCV DNN 进行 ONNX 推断。
    - `vid_stride`: 视频帧率的步长。
    - `retina_masks`: 是否使用高分辨率的掩码。
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',  type=str, default=ROOT / 'best.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / '441.jpg',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--view-img', default=True, action='store_true', help='View the prediction images')
    parser.add_argument('--data', type=str, default=ROOT / 'data/618A.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--save-txt', default=True, action='store_true',
                        help='Save the results in a txt file')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--save-conf', default=True, action='store_true', help='Save the condidence scores')
    parser.add_argument('--save-crop', default=True, action='store_true', help='Save the crop')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='Hide the labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='Hide the confidence scores')
    parser.add_argument('--vid-stride', default=False, action='store_true',
                        help='Input video frame-rate stride')
    parser.add_argument('--line-thickness', default=3, type=int, help='Bounding-box thickness (pixels)')
    parser.add_argument('--visualize', default=False, action='store_true', help='Visualize model features')
    parser.add_argument('--augment', default=False, action='store_true', help='Augmented inference')
    parser.add_argument('--agnostic-nms', default=False, action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--retina-masks', default=False, action='store_true', help='High resolution masks')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
