
schedctl create --name cascade4 --image "10.81.138.201:5000/xsemseg:v1.4" --gpu 4 --cmd "cd /data/home/jiachen/project/SMore-Det/my_exp/mmdetection && source ~/.bashrc && conda activate mmd  && python -m torch.distributed.launch --nnodes 4 --node_rank 0 --master_addr "127.0.0.1" --nproc_per_node 1 --master_port 29500 tools/train.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco4.py --seed 0 --launcher pytorch ${@:3} --work-dir output_cascade4"



