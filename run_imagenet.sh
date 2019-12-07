python -m torch.distributed.launch --nproc_per_node=4 train_groupnet_imagenet.py \
                                   -a "resnet50" \
                                   --data "/media/ssd0/ilsvrc12/rec" \
                                   --tmp "results/imagenet-res50-sparsity1e-5-thres0.05-power0.3-warmup5-group1x1" \
                                   --static-loss-scale "128.0" \
                                   --use-rec \
                                   --dali-cpu \
                                   --sparsity "1e-5" \
                                   --sparse-thres "0.05" \
                                   --power "0.3" \
                                   --warmup "5" \
                                   --group1x1 \
                                   --init-iters 20 \
                                   --epoch-iters 5 \
                                   --iter-iters 1
