import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
import torchvision
import setproctitle

import os
import sys
import math

import shutil
import my_densenet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_PATH = os.path.abspath('./')
data_path = os.path.join(ROOT_PATH, 'data')
train_data_path = os.path.join(data_path, '224', 'train')
val_data_path = os.path.join(data_path, '224', 'val')
test_data_path = os.path.join(data_path, '224', 'test')
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0, 0, 0],
                              std=[1, 1, 1]),
         ])
train_data = torchvision.datasets.ImageFolder(train_data_path, transform=transform)
val_data = torchvision.datasets.ImageFolder(val_data_path, transform=transform)
test_data = torchvision.datasets.ImageFolder(test_data_path, transform=transform)

def ausgabeMetriken(tp, tn, fp, fn, p, n):
    print("there are {} pos".format(p))
    print("there are {} neg".format(n))
    print("True positive: {}".format(tp))
    print("True negative: {}".format(tn))
    print("False positive: {}".format(fp))
    print("False negative: {}".format(fn))
    if (tp+fp) == 0:
        print("all preds are negative")
    else:
        epoch_precision = tp / (tp+fp)
        print("precision: {}".format(epoch_precision))
        fdr = fp / (fp+tp)
        print("false discovery rate: {}".format(fdr))
    if (tp+fn) == 0:
        print("no positives in input data")
    else:
        epoch_recall = tp / (tp+fn)
        fnr = fn / (fn+tp)
        print("recall, sensitivity, tpr: {}".format(epoch_recall))
        print("miss rate, false negative rate: {}".format(fnr))
    if not (2*tp+fp+fn == 0):
        f1 = 2*tp / (2*tp+fp+fn)
        print("f1-score: {}".format(f1))
    if (tn+fp != 0):
        tnr = tn /(tn+fp)
        print("specificity, tnr: {}".format(tnr))
        fpr = fp / (fp+tn)
        print("fall-out, fpr: {}".format(fpr))
    if (tn+fn != 0):
        npv = tn / (tn+fn)
        print("negative predictive value: {}".format(npv))
        falseor = fn / (fn+tn)
        print("false omission rate: {}".format(falseor))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=2)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/densenet.base'
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)
    num_workers=0
    batch_size = args.batchSz

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
         num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, 
         num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers, shuffle=False)

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    net = my_densenet.DenseNet(growthRate=1, depth=25, reduction=0.75,
                            bottleneck=True, device=device)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')
    criterion = torch.nn.BCELoss()

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        print("Training Epoch {}".format(epoch))
        train(args, epoch, net, train_loader, optimizer, criterion)
        print("Validation Epcoh{}".format(epoch))
        test(args, epoch, net, valid_loader, optimizer, criterion)
        torch.save(net, os.path.join(args.save, 'latest.pth'))
        os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()

def train(args, epoch, net, trainLoader, optimizer, criterion):
    net.train()
    nProcessed = 0
    train_loss = 0.0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    p = 0
    n = 0
    train_running_correct = 0
    nTrain = len(trainLoader.dataset)
    for data, labels in trainLoader:
        if args.cuda:
            data, labels = data.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(data)
        target = outputs.squeeze()
        labels = labels.to(torch.float32)
        t = torch.Tensor([0.5])  # threshold
        target = target.to(device)
        t = t.to(device)
        preds = (target > t).float() * 1
        train_running_correct += (preds == labels).sum().item()
        for i in range(0, len(preds)):
            if (preds[i] == 1.) & (labels[i] == 0.):
                fp += 1
            elif (preds[i] == 0.) & (labels[i] == 1.):
                fn += 1
            elif preds[i] == labels[i] == 0.:
                tn += 1
            elif preds[i] == labels[i] == 1.:
                tp += 1
            if labels[i] == 1.:
                p +=1
            else:
                n +=1
        loss = criterion(target, labels)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        incorrect = fp+fn
        err = 100.*incorrect/len(data)
        train_loss += loss.item()*data.size(0)


        #trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        #trainF.flush()
    ausgabeMetriken(tp, tn, fp, fn, p, n)
    
    train_loss = train_loss/len(trainLoader.sampler)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, train_loss))


def test(args, epoch, net, testLoader, optimizer, criterion):
    net.eval()
    test_loss = 0
    valid_loss = 0
    incorrect = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    p = 0
    n = 0
    for data, labels in testLoader:
        if args.cuda:
            data, labels = data.cuda(), labels.cuda()
        outputs = net(data)
        target = outputs.squeeze()
        labels = labels.to(torch.float32)
        t = torch.Tensor([0.5])  # threshold
        target = target.to(device)
        t = t.to(device)
        preds = (target > t).float() * 1
        for i in range(0, len(preds)):
            if (preds[i] == 1.) & (labels[i] == 0.):
                fp += 1
            elif (preds[i] == 0.) & (labels[i] == 1.):
                fn += 1
            elif preds[i] == labels[i] == 0.:
                tn += 1
            elif preds[i] == labels[i] == 1.:
                tp += 1
            if labels[i] == 1.:
                p +=1
            else:
                n +=1
        test_loss += criterion(target, labels)
        # calculate the batch loss
        loss = criterion(target, labels)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
        incorrect = fp+fn

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    valid_loss = valid_loss/len(testLoader.sampler)
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    #print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, incorrect, nTotal, err))

    #testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    #testF.flush()
    ausgabeMetriken(tp, tn, fp, fn, p, n)
    
    print('Epoch: {} \tValidation Loss: {:.6f}'.format(
        epoch, valid_loss))

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-1
        elif epoch == 150: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__=='__main__':
    main()