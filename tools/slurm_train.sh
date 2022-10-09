

schedctl create --name cascade33 --image "10.81.138.201:5000/xsemseg:v1.4" --gpu 4 --cmd "cd /data/home/jiachen/project/SMore-Det/my_exp/mmdetection && source ~/.bashrc && conda activate mmd  && --gres=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --kill-on-bad-exit=1 python -u tools/train.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py --work-dir=output_cascade33 --launcher="slurm" 1"






# schedctl create --name cascade --image "10.81.138.201:5000/xsemseg:v1.4" --gpu 4 --cmd "cd /data/home/jiachen/project/SMore-Det/my_exp/mmdetection && source ~/.bashrc && conda activate mmd  && python tools/train.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py --gpus 1 --work-dir output_cascade"


# CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh def ${JOB_NAME} config1.py ${WORK_DIR} --options 'dist_params.port=29500'

