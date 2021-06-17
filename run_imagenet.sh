CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=2345 train_imagenet.py \
                -a "resnet50" \
                -j "4" \
                --data "/mnt/truenas/scratch/hzh/datasets/imagenet_rec" \
                --save-path "results/imagenet-resnet50-prune-percent-0.35" \
                --batch-size "64" \
                --prune-percent "0.35"
