schedctl create --name inference --image "10.81.138.201:5000/xsemseg:v1.4" --gpu 1 --cmd "cd /data/home/jiachen/project/SMore-Det/my_exp/mmdetection && source ~/.bashrc && conda activate mmd  && python tools/test.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_cocoohem16.py output_cascade_ohem16/latest.pth --show --format-only --options "jsonfile_prefix=ohem32.json""