import os

import numpy as np
import torch
import torchvision
import tqdm
from ray import tune
from ray.tune import CLIReporter, register_trainable
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
from torchvision import transforms

import config
import my_tunemodel
import my_model
import training
import validate

MAX_NUM_EPOCHS = config.MAX_NUM_EPOCHS
GRACE_PERIOD = config.GRACE_PERIOD
CPU = config.CPU
GPU =config.GPU
NUM_SAMPLES = config.NUM_SAMPLES

config ={
        'num_conv':  tune.sample_from(lambda _: np.random.randint(1, 7)),
        'num_filters': tune.sample_from(lambda _: 2 ** np.random.randint(0, 6)),
        'use_bias': tune.choice([True, False]),
        'kernel_initializer': tune.choice(['random_uniform', 'glorot_uniform']),
        'kernel_size': tune.sample_from(lambda _: np.random.randint(2, 6)),
        'max_pooling': tune.choice(['none', '1th-2', '2th-2', '2th-3']),
        'stride': tune.sample_from(lambda _: np.random.randint(0, 2)),
        'bns': tune.choice(['none', '1th', '2th', '3th', '4th']),
        'num_linear': tune.sample_from(lambda _: np.random.randint(0, 7)),
        'num_units': tune.sample_from(lambda _: 2 ** np.random.randint(4, 10)),        
        'inner_act': tune.choice(['relu', 'tanh']),
        'conv_act': tune.choice(['relu', 'tanh', 'linear']),
        'linear_act': tune.choice(['relu', 'tanh', 'linear']),
        'use_bias_out': tune.choice([True, False]),
        'additonal_output_layer': tune.choice([True, False]),
        'opt': tune.choice(['adam', 'sgd']),
        'lr': tune.loguniform(1e-4, 1e-1),
        'first_conv_out': 
            tune.sample_from(lambda _: 2 ** np.random.randint(4, 8)),
        'first_fc_out': 
            tune.sample_from(lambda _: 2 ** np.random.randint(4, 8)),
        'lr': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.choice([2, 4, 8, 16]),
        'decay': tune.loguniform(1e-8, 1e-4),
        'momentum': tune.sample_from(lambda _: np.random.randint(1, 9)/10),
        'epochs': tune.sample_from(lambda _: np.random.randint(2, 15)*20),
        'metric': tune.choice(['loss']),
        'mode': tune.choice(['max']),
        'max_time': tune.sample_from(lambda _: np.random.randint(99, 101)),
        'grace_period': tune.sample_from(lambda _: np.random.randint(9, 11)),
        'reduction_factor': tune.sample_from(lambda _: np.random.randint(2, 5)),
        'report_metrics': tune.choice(["loss", "accuracy", "training_iteration"])
    }

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
dataset_train = torchvision.datasets.ImageFolder(train_data_path, transform=transform)
dataset_val = torchvision.datasets.ImageFolder(val_data_path, transform=transform)
dataset_test = torchvision.datasets.ImageFolder(test_data_path, transform=transform)

def train_and_validate(config, checkpoint_dir = None):
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = my_tunemodel.CustomNet(config['first_conv_out'], config['first_fc_out']).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    if config['opt'] == "adam":
        optimizer = torch.optim.Adam(model.parameters, lr = config['lr'], decay=config['decay'])
    elif config['opt'] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    criterion = torch.nn.BCELoss()
    EPOCHS = config['epochs']

    for epoch in range(EPOCHS):
        print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
        train_epoch_loss, train_epoch_acc = training.train(model, dataloader_train, optimizer, criterion, device)
        valid_epoch_loss, valid_epoch_acc = validate.validate(model, dataloader_val, criterion, device)
  
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'checkpoint')
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=valid_epoch_loss, accuracy=valid_epoch_acc)

def run_search():
    # Scheduler to stop bad performing trails.
    scheduler = ASHAScheduler(
        #metric=config['metric'],
        metric="loss",
        #mode=config['mode'],
        mode = "min",
        #max_t=config['max_time'],
        max_t=MAX_NUM_EPOCHS,
        #grace_period=config['grace_period'],
        grace_period=GRACE_PERIOD,
        #reduction_factor=config['reduction_factor']
        reduction_factor=2
    )

    # Reporter to show on command line/output window
    reporter = CLIReporter(
        #metric_columns=config['report_metrics']
        metric_columns=["loss", "accuracy", "training_iteration"])

    # Start run/search
    result = tune.run(
        train_and_validate,
        resources_per_trial={"cpu": CPU, "gpu": GPU},
        config=config,
        num_samples=NUM_SAMPLES,
        scheduler=scheduler,
        local_dir='./data/',
        keep_checkpoints_num=1,
        checkpoint_score_attr='min-validation_loss',
        progress_reporter=reporter
    )
    # Extract the best trial run from the search.
    best_trial = result.get_best_trial(
        'loss', 'min', 'last'
    )
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation acc: {best_trial.last_result['accuracy']}")

def __main__():
    run_search()

if __name__ == '__main__':
    __main__()