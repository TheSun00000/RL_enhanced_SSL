'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils_ import progress_bar
import torchvision.models as models


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--trainbs', default=512, type=int, help='trainloader batch size')
parser.add_argument('--testbs', default=512, type=int, help='testloader batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
args = parser.parse_args()
ngpus = torch.cuda.device_count()
cuda = torch.cuda.is_available()

if args.local_rank == 0:
    print('####################################')
    print(ngpus)

# initialize PyTorch distributed using environment variables
# (you could also do this more explicitly by specifying `rank` and `world_size`,
# but I find using environment variables makes it so that you can easily use the same script on different machines)
dist.init_process_group(backend='nccl', init_method='env://')

torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
if args.local_rank == 0:
    print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./dataset/', train=True, download=True, transform=transform_train)
sampler = DistributedSampler(trainset)
trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=args.trainbs, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./dataset/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.testbs, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.local_rank == 0:
    print('==> Building model..')
# net = VGG('VGG16')
net = models.resnet50(pretrained=False)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = ResNeXt29_32x4d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        print(x.shape)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# net = SimpleClassifier(input_size=64*64*3, hidden_size=128, num_classes=10)

net = net.to(device)
net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
if ngpus > 0:
    net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.1)

# Training
def train(epoch):
    if args.local_rank == 0:
        print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
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

        if args.local_rank == 0:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.local_rank == 0:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.


for epoch in range(start_epoch, start_epoch+300):
    # In PyTorch 1.1.0 and later,
    # you should call them in the opposite order:
    # `optimizer.step()` before `lr_scheduler.step()`
    sampler.set_epoch(epoch)
    train(epoch)
    
    exit()
    
    # test(epoch)
    scheduler.step()  # 每隔100 steps学习率乘以0.1

print("\nTesting best accuracy:", best_acc)