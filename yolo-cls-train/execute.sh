
#https://docs.ultralytics.com/tasks/classify/

InputDir='/media/maquina02/HD/Dados/Fernando/DATASET/YOLO-COPY/TESE/BER/BER2024/BER2024-BODY'

python3 main.py --dataset-dir $InputDir --model yolov8n-cls --output-dir yolov8n-cls --epochs 300 --batch-size 8
python3 main.py --dataset-dir $InputDir --model yolov8s-cls --output-dir yolov8s-cls --epochs 300 --batch-size 8
python3 main.py --dataset-dir $InputDir --model yolov8m-cls --output-dir yolov8m-cls --epochs 300 --batch-size 8
python3 main.py --dataset-dir $InputDir --model yolov8l-cls --output-dir yolov8l-cls --epochs 300 --batch-size 8
python3 main.py --dataset-dir $InputDir --model yolov8x-cls --output-dir yolov8x-cls --epochs 300 --batch-size 8
