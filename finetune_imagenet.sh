python -m torch.distributed.launch --nproc_per_node=4 finetune_imagenet.py \
                                   -a "resnet101" \
                                   --data "/media/ssd/imagenet/rec" \
                                   --tmp "results/imagenet-resnet101-delta-lambda2e-6-thres0.1-power0.5-warmup5-group1x1-wd0-percent0.65-adjust-lambda-from-scratch" \
                                   --resume "results/imagenet-resnet101-delta-lambda2e-6-thres0.1-power0.5-warmup5-group1x1-wd0-percent0.65-adjust-lambda/checkpoint.pth" \
                                   --batch-size "64" \
                                   --use-rec \
                                   --dali-cpu \
                                   --warmup "5" \
                                   --group1x1 \
				   --percent "0.65"
