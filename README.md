This code is the implementation of common attention modules for CNNs, such as CBAM, BAM, Squeeze and excitation (SE), shuffle Attention, and ECA-Net.

##### To train (Attention choices =['BAM', 'CBAM','ECAAttention', 'SEAttention','ShuffleAttention'] and dataset choices=['cifar10', 'cifar100']
python train.py --ngpu 1 --workers 4 --arch resnet --depth 50 --epochs 100 --batch-size 256 --lr 0.1 --att-type SEAttention --dataset cifar10

