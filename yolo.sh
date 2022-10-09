schedctl create --name yolo --image "10.81.138.201:5000/xsemseg:v1.4" --gpu 1 --cmd "cd /data/home/jiachen/project/SMore-Det/my_exp/mmdetection && source ~/.bashrc && conda activate mmd  && python tools/train.py configs/yolo/yolov3_d53_mstrain-608_273e_coco.py  --gpus 1 --work-dir output_yolo"

