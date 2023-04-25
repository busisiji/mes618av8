import os

import cv2

from ultralytics import YOLO

from prepare_data import convert

model = YOLO("runs/detect/train6/weights/best.pt")
success = model.export(format="ONNX")  # 将模型导出为 ONNX 格式
