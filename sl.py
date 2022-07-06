import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import wandb

from models import *

# import yappi
# import os
# import time

# from torch.optim.lr_scheduler import StepLR

TRAIN_SUBSET = 40000
TEST_SUBSET = 10000
TRAIN_BS = 4096
TEST_BS = 4096
EPOCHS = 200
TRAIN_EPOCH_ROUNDS = math.ceil(TRAIN_SUBSET / TRAIN_BS)
TEST_EPOCH_ROUNDS = math.ceil(TEST_SUBSET / TEST_BS)
TRAIN_SHUFFLE = False
TEST_SHUFFLE = False

LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

CUDA = True
PRINT_CLASS_ACCURACY = False
PRINT_ROUNDS = True
PRINT_ROUND_INTERVAL = 10


class SplitImageDataset(Dataset):
    def __init__(self, data=None, targets=None):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        if self.data is not None and self.targets is not None:
            return self.data[index], self.targets[index]
        elif self.data is not None:
            return self.data[index], torch.tensor([])
        elif self.targets is not None:
            return torch.tensor([]), self.targets[index]
        else:
            return torch.tensor([]), torch.tensor([])

    def __len__(self):
        if self.data is not None:
            return len(self.data)
        elif self.targets is not None:
            return len(self.targets)
        else:
            return 0


class SimpleCNNFront(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return x


class SimpleCNNBack(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        output = self.fc2(x)
        return output


def create_model_client(device, use_cuda):
    torch.manual_seed(42)
    if use_cuda:
        torch.cuda.manual_seed_all(42)

    model_client = ResNet18Front().to(device)
    optimizer_client = optim.SGD(model_client.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    return model_client, optimizer_client

def create_model_server(device, use_cuda):
    torch.manual_seed(42)
    if use_cuda:
        torch.cuda.manual_seed_all(42)

    model_server = ResNet18Back().to(device)
    optimizer_server = optim.SGD(model_server.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # TODO: Use Adam with StepLR
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    # scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

    return model_server, optimizer_server

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_loaders(test_kwargs, train_kwargs):
    train_transform = transforms.Compose(
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

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )

    # =====
    # Train
    # =====

    trainset = torchvision.datasets.CIFAR10(root='/tmp/cifar10_data', train=True, download=True, transform=train_transform)
    sub = Subset(trainset, np.arange(0, 40000))
    all_data = []
    all_targets = []
    for d, t in sub:
        all_data.append(d)
        all_targets.append(torch.tensor(t))
    all_data = torch.stack(all_data)
    all_targets = torch.stack(all_targets)

    # Clients Train

    trainset_client = SplitImageDataset(data=all_data)
    g = torch.Generator()
    g.manual_seed(42)
    trainloader_client = torch.utils.data.DataLoader(trainset_client.data, generator=g, **train_kwargs)
    trainloader_iter_client = iter(cycle(trainloader_client))

    # Server Train
    trainset_server = SplitImageDataset(targets=all_targets)
    g = torch.Generator()
    g.manual_seed(42)
    trainloader_server = torch.utils.data.DataLoader(trainset_server.targets, generator=g, **train_kwargs)
    trainloader_iter_server = iter(cycle(trainloader_server))

    # ====
    # Test
    # ====

    testset = torchvision.datasets.CIFAR10(root='/tmp/cifar10_data', train=True, download=True, transform=test_transform)

    sub = Subset(testset, np.arange(40000, 50000))
    all_data = []
    all_targets = []
    for d, t in sub:
        all_data.append(d)
        all_targets.append(torch.tensor(t))
    all_data = torch.stack(all_data)
    all_targets = torch.stack(all_targets)

    # Clients Test
    testset_client = SplitImageDataset(data=all_data)
    g = torch.Generator()
    g.manual_seed(0)
    testloader_client = torch.utils.data.DataLoader(testset_client.data, generator=g, **test_kwargs)
    testloader_iter_client = iter(cycle(testloader_client))

    # Server Test
    testset_server = SplitImageDataset(targets=all_targets)
    g = torch.Generator()
    g.manual_seed(0)
    testloader_server = torch.utils.data.DataLoader(testset_server.targets, generator=g, **test_kwargs)
    testloader_iter_server = iter(cycle(testloader_server))

    return (
        trainloader_iter_client,
        trainloader_iter_server,
        testloader_iter_client,
        testloader_iter_server,
    )


def forward_client(model, device, trainloader_iter, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()
    data = next(trainloader_iter).to(device)
    if optimizer:
        optimizer.zero_grad()
    activations = model(data)
    return activations, activations.detach().cpu().numpy()


def backward_client(device, optimizer, activations, gradients):
    gradients = torch.from_numpy(gradients)
    gradients = gradients.to(device)
    activations.backward(gradients)
    optimizer.step()


def train_server(model, device, trainloader_iter, optimizer, criterion, activations):
    model.train()

    activations = torch.from_numpy(activations).to(device)
    activations.requires_grad = True
    activations.retain_grad()

    target = next(trainloader_iter).to(device)
    optimizer.zero_grad()
    outputs = model(activations)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

    # _, pred = outputs.max(1)
    pred = outputs.argmax(dim=1, keepdim=True)

    return target, pred, loss.item(), activations.grad.detach().cpu().numpy()


def test_server(model, device, testloader_iter, criterion, activations):
    model.eval()

    with torch.no_grad():
        activations = torch.from_numpy(activations)
        activations = activations.to(device)

        target = next(testloader_iter).to(device)
        outputs = model(activations)
        loss = criterion(outputs, target)

        # _, pred = outputs.max(1)
        pred = outputs.argmax(dim=1, keepdim=True)

    return pred, target, loss.item()

def train(
    device,
    epoch,
    model_client,
    model_server,
    optimizer_client,
    optimizer_server,
    criterion,
    trainloader_iter_client,
    trainloader_iter_server
):
    train_loss_total = 0
    train_loss_running = 0
    train_total = 0
    train_correct = 0

    for flround in range(TRAIN_EPOCH_ROUNDS):
        activations, activations_detached = forward_client(model_client, device, trainloader_iter_client, optimizer_client)

        pred, target, loss, gradients = train_server(
            model_server,
            device,
            trainloader_iter_server,
            optimizer_server,
            criterion,
            activations_detached
        )

        train_loss_total += loss
        train_loss_running += loss
        train_total += target.size(0)
        # train_correct += pred.eq(target).sum().item()
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        train_acc = train_correct / train_total

        if (
            (flround % PRINT_ROUND_INTERVAL == PRINT_ROUND_INTERVAL - 1) or flround == TRAIN_EPOCH_ROUNDS - 1
        ) and PRINT_ROUNDS:
            print(
                "TRAIN R: {:5d} - LOSS CURRENT: {:.3f} - LOSS {:2d} LAST ROUNDS: {:.3f} - ACC: {:.1f}%".format(
                    flround + 1, loss, PRINT_ROUND_INTERVAL, train_loss_running / PRINT_ROUND_INTERVAL, 100.0 * train_acc
                )
            )
            train_loss_running = 0.0

        backward_client(device, optimizer_client, activations, gradients)

    train_loss_average = train_loss_total / TRAIN_EPOCH_ROUNDS

    print(
        "EPOCH {} TRAIN - AVERAGE LOSS: {:.3f} - ACC: {}/{} ({:.1f}%)".format(
            epoch + 1, train_loss_average, train_correct, train_total, 100.0 * train_acc
        )
    )

    metrics = {
        "train/epoch/epoch": epoch  + 1,
        "train/epoch/accuracy": train_acc,
        "train/epoch/loss": train_loss_average,
    }

    wandb.log(metrics)

    return metrics

def test(device,
         epoch,
         model_client,
         model_server,
         criterion,
         testloader_iter_client,
         testloader_iter_server
):
    test_loss_total = 0
    test_loss_running = 0
    test_total = 0
    test_correct = 0

    classes = range(10)
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    for flround in range(TEST_EPOCH_ROUNDS):
        _, activations_detached = forward_client(model_client, device, testloader_iter_client)

        pred, target, loss = test_server(
            model_server, device, testloader_iter_server, criterion, activations_detached
        )

        test_loss_total += loss
        test_loss_running += loss
        test_total += target.size(0)
        # test_correct += predictions.eq(target).sum().item()
        test_correct += pred.eq(target.view_as(pred)).sum().item()
        test_acc = test_correct / TEST_SUBSET

        if (
            (flround % PRINT_ROUND_INTERVAL == PRINT_ROUND_INTERVAL - 1) or flround == TEST_EPOCH_ROUNDS - 1
        ) and PRINT_ROUNDS:
            print(
                "TEST R: {:5d} - LOSS CURRENT: {:.3f} - LOSS {:2d} LAST ROUNDS: {:.3f} - ACC: {:.1f}%".format(
                    flround + 1, loss, PRINT_ROUND_INTERVAL, test_loss_running / PRINT_ROUND_INTERVAL, 100.0 * test_acc
                )
            )
            test_loss_running = 0

        for label, prediction in zip(target, pred):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    test_loss_average = test_loss_total / TEST_EPOCH_ROUNDS

    print(
        "EPOCH {} TEST - AVERAGE LOSS: {:.4f} - ACC: {}/{} ({:.1f}%)\n".format(
            epoch + 1, test_loss_average, test_correct, test_total, 100.0 * test_acc
        )
    )

    if PRINT_CLASS_ACCURACY:
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f"CLASS {classname} ACC: {accuracy:.1f}%")

    metrics = {
        "test/epoch/epoch": epoch + 1,
        "test/epoch/accuracy": test_acc,
        "test/epoch/loss": test_loss_average}

    return metrics


def main():
    # yappi.set_clock_type("cpu")  # Use set_clock_type("wall") for wall time
    # yappi.start()
    torch.manual_seed(42)
    use_cuda = CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": TRAIN_BS, "num_workers": 2, "shuffle": TRAIN_SHUFFLE, "worker_init_fn": seed_worker}
    test_kwargs = {"batch_size": TEST_BS, "num_workers": 2, "shuffle": TEST_SHUFFLE, "worker_init_fn": seed_worker}

    if use_cuda:
        torch.cuda.manual_seed_all(42)
        cuda_kwargs = {"pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    (
        trainloader_iter_client,
        trainloader_iter_server,
        testloader_iter_client,
        testloader_iter_server,
    ) = create_data_loaders(train_kwargs, test_kwargs)

    model_client, optimizer_client = create_model_client(device, use_cuda)
    model_server, optimizer_server = create_model_server(device, use_cuda)
    criterion = nn.CrossEntropyLoss()

    wandb.init(project="CIFAR-10-Standalone", name="CIFAR-10 - Standalone", entity="elias-manolakos-flai")

    wandb.config = {"learning_rate": LR, "epochs": EPOCHS, "batch_size": TRAIN_BS}

    for epoch in range(EPOCHS):
        train_metrics = train(
            device,
            epoch,
            model_client,
            model_server,
            optimizer_client,
            optimizer_server,
            criterion,
            trainloader_iter_client,
            trainloader_iter_server,
        )

        test_metrics = test(
            device,
            epoch,
            model_client,
            model_server,
            criterion,
            testloader_iter_client,
            testloader_iter_server
        )

        wandb.log({**train_metrics, **test_metrics})

        # scheduler.step()

    wandb.finish()

    # yappi.stop()
    # profile_path = os.path.join("./", "yappi_mnist_splitnn_{}.pstat".format(time.time()))
    # yappi.get_func_stats().save(profile_path, type="pstat") # print_all()
    # yappi.get_func_stats().print_all()
    # yappi.get_thread_stats().print_all()


if __name__ == "__main__":
    main()
