python -m torch.distributed.launch --nproc_per_node=4 trainimagenet.py \
                                   -a "resnet50" \
                                   --data "/media/ssd1/ilsvrc12/rec" \
                                   --tmp "results/imagenet-resnet50-baseline" \
                                   --batch-size "64" \
                                   --use-rec \
                                   --dali-cpu \
                                   --warmup "5"
