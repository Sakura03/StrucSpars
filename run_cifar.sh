CUDA_VISIBLE_DEVICES='0' python3 train_cifar.py -a "resnet20" --data "./data" --dataset "cifar10" --save-path "results/cifar10-resnet20-prune-percent-0.4" --prune-percent "0.4"
