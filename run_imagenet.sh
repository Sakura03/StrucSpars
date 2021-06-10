CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=2345 train_groupnet_imagenet.py \
                                -a "resnet50" \
                                --data "/path/to/rec"
                                --save-path "results/imagenet-resnet50-percent0.35" \
                                --batch-size "64" \
                                --epochs "60" \
                                --wd "0." \
                                --warmup "5" \
                                --percent "0.35" \
                                --finetune-epochs "120"
