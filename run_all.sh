#!/usr/bin/env bash

#
# ResNet
#

python3 main.py --model="resnet18" --bs=512
python3 main.py --model="resnet18" --bs=1024
python3 main.py --model="resnet18" --bs=2048
python3 main.py --model="resnet18" --bs=4096
python3 main.py --model="resnet18" --bs=8192

python3 main.py --model="resnet34" --bs=512
python3 main.py --model="resnet34" --bs=1024
python3 main.py --model="resnet34" --bs=2048
python3 main.py --model="resnet34" --bs=4096
python3 main.py --model="resnet34" --bs=8192

python3 main.py --model="resnet50" --bs=512
python3 main.py --model="resnet50" --bs=1024
python3 main.py --model="resnet50" --bs=2048
python3 main.py --model="resnet50" --bs=4096
python3 main.py --model="resnet50" --bs=8192

python3 main.py --model="resnet18" --nolrs --bs=512
python3 main.py --model="resnet18" --nolrs --bs=1024
python3 main.py --model="resnet18" --nolrs --bs=2048
python3 main.py --model="resnet18" --nolrs --bs=4096
python3 main.py --model="resnet18" --nolrs --bs=8192

python3 main.py --model="resnet34" --nolrs --bs=512
python3 main.py --model="resnet34" --nolrs --bs=1024
python3 main.py --model="resnet34" --nolrs --bs=2048
python3 main.py --model="resnet34" --nolrs --bs=4096
python3 main.py --model="resnet34" --nolrs --bs=8192

python3 main.py --model="resnet50" --nolrs --bs=512
python3 main.py --model="resnet50" --nolrs --bs=1024
python3 main.py --model="resnet50" --nolrs --bs=2048
python3 main.py --model="resnet50" --nolrs --bs=4096
python3 main.py --model="resnet50" --nolrs --bs=8192

#
# VGG
#

python3 main.py --model="vgg11" --bs=512
python3 main.py --model="vgg11" --bs=1024
python3 main.py --model="vgg11" --bs=2048
python3 main.py --model="vgg11" --bs=4096
python3 main.py --model="vgg11" --bs=8192

python3 main.py --model="vgg16" --bs=512
python3 main.py --model="vgg16" --bs=1024
python3 main.py --model="vgg16" --bs=2048
python3 main.py --model="vgg16" --bs=4096
python3 main.py --model="vgg16" --bs=8192

python3 main.py --model="vgg19" --bs=512
python3 main.py --model="vgg19" --bs=1024
python3 main.py --model="vgg19" --bs=2048
python3 main.py --model="vgg19" --bs=4096
python3 main.py --model="vgg19" --bs=8192

python3 main.py --model="vgg11" --nolrs --bs=512
python3 main.py --model="vgg11" --nolrs --bs=1024
python3 main.py --model="vgg11" --nolrs --bs=2048
python3 main.py --model="vgg11" --nolrs --bs=4096
python3 main.py --model="vgg11" --nolrs --bs=8192

python3 main.py --model="vgg16" --nolrs --bs=512
python3 main.py --model="vgg16" --nolrs --bs=1024
python3 main.py --model="vgg16" --nolrs --bs=2048
python3 main.py --model="vgg16" --nolrs --bs=4096
python3 main.py --model="vgg16" --nolrs --bs=8192

python3 main.py --model="vgg19" --nolrs --bs=512
python3 main.py --model="vgg19" --nolrs --bs=1024
python3 main.py --model="vgg19" --nolrs --bs=2048
python3 main.py --model="vgg19" --nolrs --bs=4096
python3 main.py --model="vgg19" --nolrs --bs=8192
