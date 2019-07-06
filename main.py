'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys

from models import *
from utils import progress_bar

from tensorboardX import SummaryWriter
import numpy as np
from data import *

#log file
# f=open('logs/2','w')
# sys.stdout=f

#tensorboardX
writer=SummaryWriter('logs/myNet-16-BNwithoutDP')
# train_loss_per_epoch=[]
# train_acc_per_epoch=[]
# test_loss_per_epoch=[]
# test_acc_per_epoch=[]
train_loss_per_epoch=0.0
train_acc_per_epoch=0.0
test_loss_per_epoch=0.0
test_acc_per_epoch=0.0


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataDir = "./data/train/"
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainset=TLDataset(dataDir)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

dataDir="./data/test/"
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testset=TLDataset(dataDir)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net=Net()
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
#net = ShuffleNetV2(1)
#net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer =torch.optim.Adam(net.parameters(), lr = args.lr, betas= (0.9, 0.99))

#Adjust lr
def adjust_rl(optimizer,epoch):
    lr=args.lr* (0.1 ** (epoch // 70))
    print("current lr: "+str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Training
def train(epoch):
    global train_loss_per_epoch,train_acc_per_epoch
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print(type(inputs))
        inputs, targets = inputs.to(device), targets.to(device)
        inputs=inputs.permute(0,3,2,1)
        # print("**********")
        # print(inputs.size())
        # print("***********")
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # train_loss_per_epoch.append(train_loss/(batch_idx+1))
        # train_acc_per_epoch.append(100.*correct/total)
    train_acc_per_epoch=100.*correct/total
    train_loss_per_epoch=train_loss/(batch_idx+1)

def test(epoch):
    global best_acc,test_loss_per_epoch,test_acc_per_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs=inputs.permute(0,3,2,1)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # train_loss_per_epoch.append(test_loss/(batch_idx+1))
            # train_acc_per_epoch.append(100.*correct/total)
        test_loss_per_epoch=test_loss/(batch_idx+1)
        test_acc_per_epoch=100.*correct/total

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        print('best acc %f'%acc)
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    else:
        print('best acc %f'%best_acc)


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    #writer.add_scalar('train_loss',train_loss,epoch)
    writer.add_scalars('loss',{'train':train_loss_per_epoch,'test':test_loss_per_epoch},epoch)
    writer.add_scalars('acc',{'train':train_acc_per_epoch,'test':test_acc_per_epoch},epoch)
    adjust_rl(optimizer,epoch)
