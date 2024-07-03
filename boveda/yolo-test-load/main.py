#!/usr/bin/python

from ultralytics import YOLO

# Caminho para o arquivo best.pt
model_path = '/media/fernando/Expansion/OUTPUTS/DOCTORADO2/cnn_emotion4/ber2024-body/training_validation_holdout_fine_tuning/yolov8m-cls/model_yolov8m-cls.pt'

# Carregar o modelo
model = YOLO(model_path)


# Verificar o modelo
print(model.model)
