'''Train CIFAR10 with PyTorch.'''
import os
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.models as ptmodels

import neptune.new as neptune
import numpy as np

from models import *
from utils import progress_bar


best_acc = 0  # best test accuracy


def main():
    run = neptune.init(
        project="eliaskousk/CIFAR-10-GCP",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMzRhZmQ3YS05OWFlLTQ2ODUtYTMwYS1kYmZlMTg5NGQwMDIifQ==",
    )

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    params = {
        "epochs": 200,
        "lr": args.lr, # 1e-2,
        "bs": 256, # 128,
        "num_classes": 10,
    }
    run["parameters"] = params

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # export PYTHONHASHSEED=0
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if use_cuda:
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42) # Multi-GPU

    # https://pytorch.org/docs/stable/notes/randomness.html
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # To enable the below add CUBLAS_WORKSPACE_CONFIG=:4096:8 environment variable
    # https: // docs.nvidia.com / cuda / cublas / index.html  # cublasApi_reproducibility
    # torch.use_deterministic_algorithms(True) # This includes the above

    dataloader_kwargs = {
        "batch_size": params["bs"],
        "drop_last": True,
        "num_workers": 2,
        "shuffle": True,
        "worker_init_fn": _seed_worker,
    }

    # Data
    print('==> Preparing data..')
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                          (0.2023, 0.1994, 0.2010)),
    # ])

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                          (0.2023, 0.1994, 0.2010)),
    # ])

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_kwargs)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    dataloader_kwargs["shuffle"] = False
    testloader = torch.utils.data.DataLoader(testset, **dataloader_kwargs)

    # Model
    print('==> Building model..')
    net = VGG('VGG11')
    # net = ptmodels.vgg11(pretrained=False, num_classes=10)
    # net = VGG('VGG16')
    # net = ptmodels.vgg16(pretrained=False, num_classes=10)
    # net = VGG('VGG19')
    # net = ptmodels.vgg19(pretrained=False, num_classes=10)

    # net = ResNet18()
    # net = ptmodels.resnet18(pretrained=False, num_classes=10)
    # net = ResNet34()
    # net = ptmodels.resnet34(pretrained=False, num_classes=10)
    # net = ResNet50()
    # net = ptmodels.resnet150(pretrained=False, num_classes=10)

    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        global best_acc
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=params["lr"], momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])

    for epoch in range(start_epoch, start_epoch + params["epochs"]):
        train(run, epoch, device, trainloader, net, criterion, optimizer)
        test(run, epoch, device, testloader, net, criterion)
        scheduler.step()

    run.stop()

def train(run, epoch, device, trainloader, net, criterion, optimizer):
    print('\nEpoch: %d' % (epoch + 1))
    train_loss = 0
    correct = 0
    total = 0
    acc = 0.0

    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = correct / total

        # run["train/batch/loss"].log(loss)
        # run["train/batch/acc"].log(acc)
        # run["train/batch/batch"].log(batch_idx + 1)
        # run["train/batch/epoch"].log(epoch + 1)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    run["train/epoch/loss"].log(train_loss)
    run["train/epoch/acc"].log(acc)
    run["train/epoch/epoch"].log(epoch + 1)

def test(run, epoch, device, testloader, net, criterion):
    global best_acc
    test_loss = 0
    correct = 0
    total = 0

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100.*correct / total, correct, total))

    acc = correct / total
    run["valid/loss"].log(test_loss)
    run["valid/acc"].log(acc)
    run["valid/epoch"].log(epoch + 1)

    # Save checkpoint.
    acc = 100.*correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    main()
