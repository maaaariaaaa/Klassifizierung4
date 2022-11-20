import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import earlystopping
from earlystopping import EarlyStopping
#from ray import tune
#from ray.tune import CLIReporter, register_trainable
#from ray.tune.schedulers import ASHAScheduler

import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_PATH = os.path.abspath('./')
data_path = os.path.join(ROOT_PATH, 'data')
train_data_path = os.path.join(data_path, 'processed', 'train')
val_data_path = os.path.join(data_path, 'processed', 'val')
test_data_path = os.path.join(data_path, 'processed', 'test')
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0, 0, 0],
                              std=[1, 1, 1]),
         ])
train_data = torchvision.datasets.ImageFolder(train_data_path, transform=transform)
val_data = torchvision.datasets.ImageFolder(val_data_path, transform=transform)
test_data = torchvision.datasets.ImageFolder(test_data_path, transform=transform)

# For search run (Ray Tune settings).
CPU = 1
GPU = 1
# Number of random search experiments to run.
NUM_SAMPLES = 20



num_train = len(train_data)
indices = list(range(num_train))


def get_suggested_params(trial):
    return {
        'opt': trial.sugggest_categorical("optimizer", ['adam', 'sgd']),
        'lr': trial.suggest_loguniform('learning-rate', 1e-5, 1e-1),
        'batch_size': trial.sugggest_categorical('batch_size', [8, 16, 24, 32]),
        'decay': trial.suggest_categorical('decay', 1e-8, 1e-4),
        'momentum': trial.suggest_float('momentum', 0.1, 0.9, step=0.1),
        'epochs': trial.suggest_int('epochs', 20, 300, step=20),
        'metric': trial.suggest_categorical(['loss']),
        'reduction_factor': trial.suggest_int('rf', 2, 5),
        'report_metrics': trial.suggest_categorical(["loss", "accuracy", "training_iteration"]),
        'hidden_layers': trial.suggest_int('hidden_layers', 128, 1024, step = [128, 256, 512, 1024]),
        'dropout': trial.suggest_float('dropout', 0.2, 0.8, step=0.1)
    }


class MyObjective(object):
    def __init__(self):
        pass


    def __call__(self, trial):
        params = get_suggested_params(trial)

        num_workers = 0
        batch_size = trial.params['batch_size']
        
        # prepare data loaders (combine dataset and sampler)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
            num_workers=num_workers, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, 
            num_workers=num_workers, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
            num_workers=num_workers, shuffle=True)


def objective(trial):

    num_workers = 0
    batch_size = config['batch_size']



    model = torchvision.models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    Hs = config['hidden_layers']
    dropout = config['dropout']
    
    model.classifier = nn.Sequential(nn.Linear(1024, Hs),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(Hs, 1),
                                 nn.Sigmoid())

    criterion = nn.BCELoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=config['lr'], momentum=config['momentum'])
    if config['opt'] == "adam":
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr = config['lr'], decay=config['decay'])
    elif config['opt'] == "sgd":
        optimizer = torch.optim.SGD(model.classifier.parameters(), lr=config['lr'], momentum=config['momentum'])

    model.to(device)
    train_on_gpu = torch.cuda.is_available()

    n_epochs = config['epochs']

    valid_loss_min = np.Inf # track change in validation loss

    early_stopping = EarlyStopping(patience=config['patience'], verbose=True)

    for epoch in range(1, n_epochs+1):

       # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        train_running_correct = 0
        fp = 0
        fn = 0
        tp = 0
        tn = 0
        p = 0
        n = 0
    
        ###################
        # train the model #
        ###################
        print("Training")
        model.train()
        for data, labels in train_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, labels = data.cuda(), labels.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(data)
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
            # calculate the batch loss
            loss = criterion(target, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
    
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
        
        
        ######################    
        # validate the model #
        ######################
        print("Validation")
        fp = 0
        fn = 0
        tp = 0
        tn = 0
        p = 0
        n = 0
        val_running_correct = 0
        model.eval()
        for data, labels in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, labels = data.cuda(), labels.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(data)
            target = outputs.squeeze()
            labels = labels.to(torch.float32)
            t = torch.Tensor([0.5])  # threshold
            target = target.to(device)
            t = t.to(device)
            preds = (target > t).float() * 1
            val_running_correct += (preds == labels).sum().item()
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
            # calculate the batch loss
            loss = criterion(target, labels)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
    
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
    
        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        train_acc = 100. * (train_running_correct / len(train_loader.dataset))
        val_acc = 100. * (val_running_correct / len(valid_loader.dataset))
        print("Training Accuracy: {}".format(train_acc))
        print("Validation Accuracy: {}".format(val_acc))
        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
    
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            try:
                torch.save(model.state_dict(), 'model_waste.pt')
                print(f'Model saved successfully to model_waste.pt')
            except:
                print(f'ERROR saving model!')
            valid_loss_min = valid_loss
    
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    trial.report(valid_loss, )
    return valid_loss
    
    

sampler = optuna.samplers.TPESampler(n_startup_trials=25)
pruner = optuna.pruners.HyperbandPruner()

study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials = 30, show_progress_bar=True)

best_trial = study.best_trial
for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))