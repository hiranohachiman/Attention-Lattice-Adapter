DATE=`date '+%Y-%m-%d-%H:%M:%S'`

python train_net.py --config-file configs/san_clip_vit_res4_coco.yaml --num-gpus 1 OUTPUT_DIR output/$DATE
