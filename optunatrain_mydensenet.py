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
import optuna
import joblib
import utils
from earlystopping import EarlyStopping

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
save_path = os.path.join(data_path, 'work', 'densenet.base')
save_study = os.path.join(data_path, 'study')
utils.create_dirs(save_path)
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

def get_suggested_params(trial):
    return {
        'opt': trial.suggest_categorical("opt", ['adam', 'sgd', 'rmsprop']),
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 24, 32]),
        'decay': trial.suggest_loguniform('decay', 1e-8, 1e-4),
        'momentum': trial.suggest_float('momentum', 0.1, 0.9, step=0.1),
        'nEpochs': trial.suggest_int('nEpochs', 20, 300, step=20),
        'reduction_factor': trial.suggest_int('rf', 2, 5),
        'report_metrics': trial.suggest_categorical('report_metrics', ["loss", "accuracy", "training_iteration"]),
        'dropout': trial.suggest_float('dropout', 0.2, 0.8, step=0.1),
        'growthRate': trial.suggest_int('growthRate', 2, 12, step=2),
        'depth': trial.suggest_int('depth', 25, 100, step=25),
        'reduction': trial.suggest_float('reduction', 0.25, 0.75, step = 0.25),
        'bottleneck':trial.suggest_categorical('bottleneck', [True, False])
    }


class MyObjective(object):
    def __init__(self):
        pass


    def __call__(self, trial):
        params = get_suggested_params(trial)
        setproctitle.setproctitle(save_path)
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)
        

        num_workers = 0
        batch_size = trial.params['batch_size']
        nEpochs = trial.params['nEpochs']


        
        # prepare data loaders (combine dataset and sampler)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
         num_workers=num_workers, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, 
            num_workers=num_workers, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
            num_workers=num_workers, shuffle=False)
        
        net = my_densenet.DenseNet(growthRate=trial.params['growthRate'], depth=trial.params['depth'], reduction=trial.params['reduction'],
                            bottleneck=trial.params['bottleneck'], device=device).to(device)
        
        print('  + Number of params: {}'.format(sum([p.data.nelement() for p in net.parameters()])))

        if trial.params['opt'] == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=trial.params['lr'],
                                momentum=trial.params['momentum'], weight_decay=trial.params['decay'])
        elif trial.params['opt']:
            optimizer = optim.Adam(net.parameters(), weight_decay=trial.params['decay'])
        elif trial.params['opt'] == 'rmsprop':
            optimizer = optim.RMSprop(net.parameters(), weight_decay=trial.params['decay'])
        
        criterion = torch.nn.BCELoss()
        
        early_stopping = EarlyStopping(patience=15, verbose=True)
        
        for epoch in range(1, nEpochs + 1):
            adjust_opt(trial.params['opt'], optimizer, epoch, trial.params['lr'])
            print("Training Epoch {}".format(epoch))
            train(epoch, net, train_loader, optimizer, criterion)
            print("Validation Epcoh{}".format(epoch))
            valid_loss = valid(epoch, net, valid_loader, criterion)
            torch.save(net, os.path.join(save_path, 'latest.pth'))
            os.system('./plot.py {} &'.format(save_path))
            early_stopping(valid_loss, net)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break
    
        return valid_loss



def train(epoch, net, trainLoader, optimizer, criterion):
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
        data, labels = data.to(device), labels.to(device)
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
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        incorrect = fp+fn
        err = 100.*incorrect/len(data)
        train_loss += loss.item()*data.size(0)

    ausgabeMetriken(tp, tn, fp, fn, p, n)
    
    train_loss = train_loss/len(trainLoader.sampler)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, train_loss))


def valid(epoch, net, testLoader, criterion):
    net.eval()
    valid_loss = 0
    incorrect = 0
    test_loss = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    p = 0
    n = 0
    for data, labels in testLoader:
        if torch.cuda.is_available():
            labels, data = labels.to(device), data.to(device)
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


    ausgabeMetriken(tp, tn, fp, fn, p, n)
    
    print('Epoch: {} \tValidation Loss: {:.6f}'.format(
        epoch, valid_loss))
    
    return valid_loss

def adjust_opt(optAlg, optimizer, epoch, paramlr):
    if optAlg == 'sgd':
        if epoch < 150: lr = paramlr/10
        elif epoch == 150: lr = lr/10
        elif epoch == 225: lr = lr/10
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def __main__():
    sampler = optuna.samplers.TPESampler(n_startup_trials=25)
    pruner = optuna.pruners.HyperbandPruner()

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(MyObjective(), n_trials = 30, show_progress_bar=True)

    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))
        return
    
    joblib.dump(study, "study.pkl")
    return

if __name__=='__main__':
    __main__()