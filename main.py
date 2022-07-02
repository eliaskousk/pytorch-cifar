'''Train CIFAR10 with PyTorch.'''
import os
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset

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
        capture_stdout=False,
        capture_stderr=False
    )

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default="resnet18", type=str, help='model')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--nolrs', '-nolrs', action='store_true', help='do not use the learning rate scheduler')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    params = {
        "model": args.model,
        "epochs": args.epochs,
        "bs": args.bs,
        "lr": args.lr,
        "use lr scheduler": not args.nolrs,
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
        "drop_last": False,
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
    dataloader_kwargs["shuffle"] = False
    trainloader = torch.utils.data.DataLoader(Subset(trainset, range(0, 40000)), **dataloader_kwargs)

    testset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_test)
    dataloader_kwargs["shuffle"] = False
    testloader = torch.utils.data.DataLoader(Subset(testset, range(40000, 50000)), **dataloader_kwargs)

    # Model
    print(f'==> Building model {args.model}..')
    if args.model == "simplecnn":
        net = SimpleCNN()
    elif args.model == "moderatecnn":
        net = ModerateCNN()
    elif args.model == "vgg11":
        net = VGG('VGG11')
    elif args.model == "pt-vgg11":
        net = ptmodels.vgg11(pretrained=False, num_classes=10)
    elif args.model == "vgg16":
        net = VGG('VGG16')
    elif args.model == "pt-vgg16":
        net = ptmodels.vgg16(pretrained=False, num_classes=10)
    elif args.model == "vgg19":
        net = VGG('VGG19')
    elif args.model == "pt-vgg19":
        net = ptmodels.vgg19(pretrained=False, num_classes=10)
    elif args.model == "resnet18":
        net = ResNet18()
    elif args.model == "pt-resnet18":
        net = ptmodels.resnet18(pretrained=False, num_classes=10)
    elif args.model == "resnet34":
        net = ResNet34()
    elif args.model == "pt-resnet34":
        net = ptmodels.resnet34(pretrained=False, num_classes=10)
    elif args.model == "resnet50":
        net = ResNet50()
    elif args.model == "pt-resnet50":
        net = ptmodels.resnet50(pretrained=False, num_classes=10)
    else:
        print(f"Model {args.model} is not supported, will now exit.")
        exit(0)

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
    if not args.nolrs:
        print("Will use the CosineAnnealingLR learning rate scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])

    for epoch in range(start_epoch, start_epoch + params["epochs"]):
        train(run, epoch, device, trainloader, net, criterion, optimizer)
        test(run, epoch, device, testloader, net, criterion)
        if not args.nolrs:
            scheduler.step()

    run.stop()

def train(run, epoch, device, trainloader, net, criterion, optimizer):
    print('\nEpoch: %d' % (epoch + 1))
    train_loss = 0
    correct = 0
    total = 0
    acc = 0.0
    loss_per_batch = 0.0

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
        loss_per_batch = train_loss / (batch_idx + 1)

        # run["train/batch/loss"].log(loss)
        # run["train/batch/acc"].log(acc)
        # run["train/batch/batch"].log(batch_idx + 1)
        # run["train/batch/epoch"].log(epoch + 1)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (loss_per_batch, 100. * acc, correct, total))

    run["train/epoch/loss"].log(loss_per_batch)
    run["train/epoch/acc"].log(acc)
    run["train/epoch/epoch"].log(epoch + 1)

def test(run, epoch, device, testloader, net, criterion):
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    acc = 0.0
    loss_per_batch = 0.0

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
            acc = correct / total
            loss_per_batch = test_loss / (batch_idx + 1)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (loss_per_batch, 100.* acc, correct, total))

    run["valid/loss"].log(loss_per_batch)
    run["valid/acc"].log(acc)
    run["valid/epoch"].log(epoch + 1)

    # Save checkpoint.
    acc = 100. * acc
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
