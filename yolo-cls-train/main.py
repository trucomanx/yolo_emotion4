#!/usr/bin/python

import os
import sys
import shutil
from ultralytics import YOLO


dataset_path='/media/maquina02/HD/Dados/Fernando/DATASET/YOLO-COPY/TESE/BER/BER2024/BER2024-BODY';
model_type='yolov8n-cls';
epochs=300;
batch_size=8;
imgsz=224;
output_dir=model_type;

for n in range(len(sys.argv)):
    if sys.argv[n]=='--dataset-dir':
        dataset_path=sys.argv[n+1];
    elif sys.argv[n]=='--model':
        model_type=sys.argv[n+1];
    elif sys.argv[n]=='--epochs':
        epochs=int(sys.argv[n+1]);
    elif sys.argv[n]=='--batch-size':
        batch_size=int(sys.argv[n+1]);
    elif sys.argv[n]=='--imgsz':
        imgsz=int(sys.argv[n+1]);
    elif sys.argv[n]=='--output-dir':
        output_dir=sys.argv[n+1];

print('dataset_path',dataset_path);
print('  model_type',model_type);
print('      epochs',epochs);
print('  batch_size',batch_size);
print('       imgsz',imgsz);
print('  output_dir',output_dir);

#os.environ["CUDA_VISIBLE_DEVICES"] = "1";

os.makedirs(output_dir,exist_ok=True);

if os.path.exists('runs/classify/train/weights'):
    # 
    print('RESUME!!!!!');
    ## https://docs.ultralytics.com/modes/train/#apple-m1-and-m2-mps-training
    # Load a model
    model = YOLO('runs/classify/train/weights/last.pt');  # load a partially trained model
    # Resume training
    results = model.train(resume=True);
else:
    # https://docs.ultralytics.com/tasks/classify/#models
    model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

    # https://docs.ultralytics.com/modes/train/#train-settings
    results = model.train(data=dataset_path, epochs=epochs, imgsz=imgsz,batch=batch_size,save=True);

    print('RESULTS:');
    print(results);
    print('END');

def verify_categorical_accuracy(model,input_dir,labels_dic={0: 'negative', 1: 'neutro', 2: 'pain', 3: 'positive'}):
    L=0;
    Count=0;
    print('Working on',input_dir);
    for ID, LABEL in labels_dic.items():
        dir_path=os.path.join(input_dir,LABEL);
        print('Working on subdirectory',LABEL,'...')
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

##############################################################################################

shutil.move('runs', output_dir);

data=dict();
data['train_categorical_accuracy']=train_acc;
data['val_categorical_accuracy']  =val_acc;
data['test_categorical_accuracy'] =test_acc;
data['train_length']=L_train;
data['val_length']=L_val;
data['test_length']=L_test;
print('data',data);

import json
with open(os.path.join(output_dir,'training_data_results.json'), "w") as write_file:
    json.dump(data, write_file, indent=4)
