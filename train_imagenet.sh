# python -m torch.distributed.launch --nproc_per_node=2 train_imagenet.py \
#                                    -a "densenet201" \
#                                    --data "/media/ssd/imagenet/rec" \
#                                    --tmp "results/imagenet-densenet-baseline" \
#                                    --batch-size "32" \
#                                    --warmup "5"

### train lightweight model
python -m torch.distributed.launch --nproc_per_node=4 train_imagenet.py \
                                   -a "shufflenet_v1" \
                                   --data "/media/ssd/imagenet/rec" \
                                   --tmp "results/imagenet-shufflenet-v1-baseline-again" \
                                   --batch-size "160" \
                                   --wd "4e-5" \
                                   --epochs "240" \
                                   --lr "0.1" \
                                   --warmup "5"
