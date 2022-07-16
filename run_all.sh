#!/usr/bin/env bash

python3 main.py --model="vgg19" --nolrs --bs=4096
#python3 main.py --model="vgg19" --nolrs --bs=2048
#python3 main.py --model="vgg19" --nolrs --bs=1024

python3 main.py --model="vgg19" --bs=4096
#python3 main.py --model="vgg19" --bs=2048
#python3 main.py --model="vgg19" --bs=1024

echo "Shutdown Will Initiate In 60 seconds!"
sleep 60
echo "Shutdown Will Happen In 60 seconds!"
sudo shutdown
echo "Execute 'sudo shutdown -c' to cancel!"

exit

#
# SimpleCNN & ModerateCNN
#

python3 main.py --model="simplecnn" --bs=128
python3 main.py --model="simplecnn" --bs=256
python3 main.py --model="simplecnn" --bs=512
python3 main.py --model="simplecnn" --bs=1024
python3 main.py --model="simplecnn" --bs=2048
python3 main.py --model="simplecnn" --bs=4096

python3 main.py --model="moderatecnn" --bs=128
python3 main.py --model="moderatecnn" --bs=256
python3 main.py --model="moderatecnn" --bs=512
python3 main.py --model="moderatecnn" --bs=1024
python3 main.py --model="moderatecnn" --bs=2048
python3 main.py --model="moderatecnn" --bs=4096

python3 main.py --model="simplecnn" --nolrs --bs=128
python3 main.py --model="simplecnn" --nolrs --bs=256
python3 main.py --model="simplecnn" --nolrs --bs=512
python3 main.py --model="simplecnn" --nolrs --bs=1024
python3 main.py --model="simplecnn" --nolrs --bs=2048
python3 main.py --model="simplecnn" --nolrs --bs=4096

python3 main.py --model="moderatecnn" --nolrs --bs=128
python3 main.py --model="moderatecnn" --nolrs --bs=256
python3 main.py --model="moderatecnn" --nolrs --bs=512
python3 main.py --model="moderatecnn" --nolrs --bs=1024
python3 main.py --model="moderatecnn" --nolrs --bs=2048
python3 main.py --model="moderatecnn" --nolrs --bs=4096

echo "Shutdown Will Initiate In 60 seconds!"
sleep 60
echo "Shutdown Will Happen In 60 seconds!"
sudo shutdown
echo "Execute 'sudo shutdown -c' to cancel!"

exit

#
# ResNet
#

python3 main.py --model="resnet18" --bs=128
python3 main.py --model="resnet18" --bs=256
python3 main.py --model="resnet18" --bs=512
python3 main.py --model="resnet18" --bs=1024
python3 main.py --model="resnet18" --bs=2048
python3 main.py --model="resnet18" --bs=4096

python3 main.py --model="resnet34" --bs=128
python3 main.py --model="resnet34" --bs=256
python3 main.py --model="resnet34" --bs=512
python3 main.py --model="resnet34" --bs=1024
python3 main.py --model="resnet34" --bs=2048
python3 main.py --model="resnet34" --bs=4096

python3 main.py --model="resnet50" --bs=128
python3 main.py --model="resnet50" --bs=256
python3 main.py --model="resnet50" --bs=512
python3 main.py --model="resnet50" --bs=1024
python3 main.py --model="resnet50" --bs=2048
python3 main.py --model="resnet50" --bs=4096

python3 main.py --model="resnet18" --nolrs --bs=128
python3 main.py --model="resnet18" --nolrs --bs=256
python3 main.py --model="resnet18" --nolrs --bs=512
python3 main.py --model="resnet18" --nolrs --bs=1024
python3 main.py --model="resnet18" --nolrs --bs=2048
python3 main.py --model="resnet18" --nolrs --bs=4096

python3 main.py --model="resnet34" --nolrs --bs=128
python3 main.py --model="resnet34" --nolrs --bs=256
python3 main.py --model="resnet34" --nolrs --bs=512
python3 main.py --model="resnet34" --nolrs --bs=1024
python3 main.py --model="resnet34" --nolrs --bs=2048
python3 main.py --model="resnet34" --nolrs --bs=4096

python3 main.py --model="resnet50" --nolrs --bs=128
python3 main.py --model="resnet50" --nolrs --bs=256
python3 main.py --model="resnet50" --nolrs --bs=512
python3 main.py --model="resnet50" --nolrs --bs=1024
python3 main.py --model="resnet50" --nolrs --bs=2048
python3 main.py --model="resnet50" --nolrs --bs=4096

#
# VGG
#

python3 main.py --model="vgg11" --bs=128
python3 main.py --model="vgg11" --bs=256
python3 main.py --model="vgg11" --bs=512
python3 main.py --model="vgg11" --bs=1024
python3 main.py --model="vgg11" --bs=2048
python3 main.py --model="vgg11" --bs=4096

python3 main.py --model="vgg16" --bs=128
python3 main.py --model="vgg16" --bs=256
python3 main.py --model="vgg16" --bs=512
python3 main.py --model="vgg16" --bs=1024
python3 main.py --model="vgg16" --bs=2048
python3 main.py --model="vgg16" --bs=4096

python3 main.py --model="vgg19" --bs=128
python3 main.py --model="vgg19" --bs=256
python3 main.py --model="vgg19" --bs=512
python3 main.py --model="vgg19" --bs=1024
python3 main.py --model="vgg19" --bs=2048
python3 main.py --model="vgg19" --bs=4096

python3 main.py --model="vgg11" --nolrs --bs=128
python3 main.py --model="vgg11" --nolrs --bs=256
python3 main.py --model="vgg11" --nolrs --bs=512
python3 main.py --model="vgg11" --nolrs --bs=1024
python3 main.py --model="vgg11" --nolrs --bs=2048
python3 main.py --model="vgg11" --nolrs --bs=4096

python3 main.py --model="vgg16" --nolrs --bs=128
python3 main.py --model="vgg16" --nolrs --bs=256
python3 main.py --model="vgg16" --nolrs --bs=512
python3 main.py --model="vgg16" --nolrs --bs=1024
python3 main.py --model="vgg16" --nolrs --bs=2048
python3 main.py --model="vgg16" --nolrs --bs=4096

python3 main.py --model="vgg19" --nolrs --bs=128
python3 main.py --model="vgg19" --nolrs --bs=256
python3 main.py --model="vgg19" --nolrs --bs=512
python3 main.py --model="vgg19" --nolrs --bs=1024
python3 main.py --model="vgg19" --nolrs --bs=2048
python3 main.py --model="vgg19" --nolrs --bs=4096

echo "Shutdown Will Initiate In 60 seconds!"
sleep 60
echo "Shutdown Will Happen In 60 seconds!"
sudo shutdown
echo "Execute 'sudo shutdown -c' to cancel!"
