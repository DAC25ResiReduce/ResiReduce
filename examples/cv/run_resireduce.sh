#!/bin/bash

# synchronous
# scp -r 

echo " cd /ResiReduce "
cd /ResiReduce


models=${1:-"resnet50"}
# models=${1:-vgg19}

# 
# VGG-19
# residual=${2:-"residual"}
# residual=${2:-"resireducevgg19"}
# residual=${2:-"none"}
# 


# 
# ResNet-50
# residual=${2:-"residual"}
residual=${2:-"resireducern50"}
# residual=${2:-"none"}
# 



##### ResiReduce-(w+2d) for VGG19
# residual=${2:-"resireducevgg19"}

##### ResiReduce-2d
# residual=${2:-"dimcprs2d"}

##### ResiReduce-a for ResNet50
# residual=${2:-"residualrn50avg2"}
##### ResiReduce-w for ResNet50
# residual=${2:-"residualrn50avg2weight"}

##### ResiReduce-a for VGG19
# residual=${2:-"residualvgg19avg"}
##### ResiReduce-w for VGG19
# residual=${2:-"residualvgg19avgweight"}

epochs=${epochs:-80}

compressor=${compressor:-"topk"}
# compressor=${compressor:-"dgc"}
# compressor=${compressor:-"qsgd"}
# compressor=${compressor:-"none"}

density=${density:-0.01}

# 打印变量值（可用于调试）
echo "Model: $models"
echo "Epochs: $epochs"
echo "Residual: $residual"
echo "Compressor: $compressor"
echo "Density: $density"


# bash examples/cv_examples/run_reuse.sh

# CMD="HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun  -np 4 -H n15:2,n16:2  "
# CMD="HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun  -np 6 -H n18:2,n16:2,n19:2  "
CMD="HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun  -np 8 -H n18:2,n16:2,n19:2,n15:2  "


# 
# 根据 models 执行不同的命令
if [ "$models" = "resnet50" ]; then
    echo "ResNet50"
    CMD+=" python examples/cv/cifar100_resnet50_topk_resireduce.py "

elif [ "$models" = "vgg19" ]; then
    echo "VGG19"
    CMD+=" python examples/cv/cifar100_vgg19_topk_resireduce.py "

else
    echo "Model not recognized. Using default command."
fi

CMD+=" --memory $residual  --epochs $epochs  --compressor $compressor --density $density "


echo "Executing: $CMD"
eval $CMD






