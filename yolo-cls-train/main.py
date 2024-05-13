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


def verify_categorical_accuracy(model,input_dir,labels_dic={0: 'negative', 1: 'neutro', 2: 'pain', 3: 'positive'}):
    L=0;
    Count=0;
    for ID, LABEL in labels_dic.items():
        dir_path=os.path.join(input_dir,LABEL);
        res=model(dir_path);
        L=L+len(res);
        for dat in res:
            if ID==dat.probs.top1:
                Count=Count+1;
    return Count*1.0/L, L;    
        
model=model.load('runs/classify/train/weights/best.pt');

train_acc, L_train = verify_categorical_accuracy(model,os.path.join(dataset_path,'train'));
val_acc  , L_val   = verify_categorical_accuracy(model,os.path.join(dataset_path,'val'));
test_acc , L_test  = verify_categorical_accuracy(model,os.path.join(dataset_path,'test'));

data=dict();
data['train_categorical_accuracy']=train_acc;
data['val_categorical_accuracy']  =val_acc;
data['test_categorical_accuracy'] =test_acc;
data['train_length']=L_train;
data['val_length']=L_val;
data['test_length']=L_test;

import json
with open("statistics.json", "w") as write_file:
    json.dump(student, write_file, indent=4)
