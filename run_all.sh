#!/usr/bin/env bash

python3 main.py --model="vgg11"
python3 main.py --model="vgg11" --bs=128
python3 main.py --model="pt-vgg11"
python3 main.py --model="pt-vgg11" --bs=128
python3 main.py --model="vgg16"
python3 main.py --model="vgg16" --bs=128
python3 main.py --model="pt-vgg16"
python3 main.py --model="pt-vgg16" --bs=128
python3 main.py --model="vgg19"
python3 main.py --model="vgg19" --bs=128
python3 main.py --model="pt-vgg19"
python3 main.py --model="pt-vgg19" --bs=128

python3 main.py --model="resnet18"
python3 main.py --model="resnet18" --bs=128
python3 main.py --model="pt-resnet18"
python3 main.py --model="pt-resnet18" --bs=128
python3 main.py --model="resnet34"
python3 main.py --model="resnet34" --bs=128
python3 main.py --model="pt-resnet34"
python3 main.py --model="pt-resnet34"  --bs=128
python3 main.py --model="resnet50"
python3 main.py --model="resnet50" --bs=128
python3 main.py --model="pt-resnet50"
python3 main.py --model="pt-resnet50" --bs=128
