#!/usr/bin/python
# export CUDA_VISIBLE_DEVICES=1
from ultralytics import YOLO

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dataset_path='/media/fernando/Expansion/DATASET/YOLO-COPY/TESE/BER/BER2024/BER2024-BODY';

if os.path.exists('runs/classify/train/weights'):
    ## https://docs.ultralytics.com/modes/train/#apple-m1-and-m2-mps-training
    # Load a model
    model = YOLO('runs/classify/train/weights/last.pt')  # load a partially trained model
    # Resume training
    results = model.train(resume=True)
else:
    # https://docs.ultralytics.com/tasks/classify/#models
    model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

    # https://docs.ultralytics.com/modes/train/#train-settings
    results = model.train(data=dataset_path, epochs=100, imgsz=224,batch=1,save=True)

    print(results)


