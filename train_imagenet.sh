python -m torch.distributed.launch --nproc_per_node=2 train_imagenet.py \
                                   -a "densenet201" \
                                   --data "/media/ssd0/ilsvrc12/rec" \
                                   --tmp "results/imagenet-densenet-baseline" \
                                   --batch-size "32" \
                                   --use-rec \
                                   --dali-cpu \
                                   --warmup "5"
