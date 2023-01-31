from tqdm import tqdm
import os
from typing import Any, Callable
import hydra
from omegaconf import OmegaConf, DictConfig
from filelock import FileLock

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import ray
from ray import tune
from ray.air import session
from ray.air.result import Result
from ray.air.config import RunConfig
from ray.tune import CLIReporter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter, Searcher
from ray.tune.schedulers import ASHAScheduler
from functools import partial

import myutils
import mymodels

import neptune.new as neptune
from neptune.new.metadata_containers.run import Run


def common_step(net: nn.Module, inputs: Tensor, labels: Tensor,
                device: torch.device, optimizer: optim.Optimizer, criterion: nn.Module) -> tuple[Tensor, Tensor]:
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs: Tensor = net(inputs)
    loss: Tensor = criterion(outputs, labels)
    return outputs, loss

def test_step(net: nn.Module, inputs: Tensor, labels: Tensor,
              device: torch.device, optimizer: optim.Optimizer, criterion: nn.Module) -> tuple[float, Tensor, int]:
    with torch.no_grad():
        outputs, loss = common_step(net, inputs, labels, device, optimizer, criterion)
        loss_value: float = loss.item()                          
        _, predicted = torch.max(outputs.data, 1)
        correct: int = (predicted==labels.to(device)).sum().item()
    return loss_value, predicted, correct

def train_step(net: nn.Module, inputs: Tensor, labels: Tensor,
               device: torch.device, optimizer: optim.Optimizer, criterion: nn.Module) -> tuple[float, Tensor, int]:
    outputs, loss = common_step(net, inputs, labels, device, optimizer, criterion)
    loss.backward()
    optimizer.step()
    loss_value: float = loss.item()
    _, predicted = torch.max(outputs.data, 1)            
    correct: int = (predicted==labels.to(device)).sum().item()
    return loss_value, predicted, correct

def train_loop(loader: DataLoader, net: nn.Module, device: torch.device, optimizer: optim.Optimizer,
               criterion: nn.Module, lr_scheduler: Any=None, train: bool=True, use_tqdm: bool=True) -> tuple[float, float]:
    sum_loss: float = 0.0; sum_correct: float = 0; sum_total: float = 0
    step: Callable = train_step if train else test_step
    if not train: net.eval()
    loader_loop: DataLoader = tqdm(loader) if use_tqdm else loader
    for (inputs, labels) in loader_loop:
        loss_value, _, correct = step(net, inputs, labels, device, optimizer, criterion)
        sum_loss += (loss_value * labels.size(0))
        sum_total += labels.size(0)
        sum_correct += correct
    if train: lr_scheduler.step()
    if not train: net.train()
    now_train_loss: float = sum_loss/sum_total
    now_train_acc: float = sum_correct/float(sum_total)
    return now_train_loss, now_train_acc

def train(configs: dict[str, Any], options: dict[str, Any], api_token: str=None,
          ) -> tuple[nn.Module, optim.Optimizer, list[float], list[float], list[float], list[float]]:
    # neptune run initialilzation
    if api_token is not None:
        run: Run = neptune.init_run(
            project="akisatok/neptune-hydra-ray-test",
            api_token=api_token,
        )
        # set options to the run
        run['options'] = options
        run['configs'] = configs
    # load data
    with FileLock(options['dataset']['dir']+'.lock'):
        trainvalset, trainset, valset, testset = myutils.load_data(
            options['dataset']['dir'], train_val_ratio=options['dataset']['train_val_ratio'])
    options['dataset']['train'] = eval(options['dataset']['train'])
    options['dataset']['val']   = eval(options['dataset']['val'])
    options['dataset']['test']  = eval(options['dataset']['test'])
    # data loader
    trainloader: DataLoader = DataLoader(
        options['dataset']['train'],
        batch_size=configs['batch_size'],
        num_workers=options['num_workers'], shuffle=True)
    valloader: DataLoader = DataLoader(
        options['dataset']['val'],
        batch_size=configs['batch_size'],
        num_workers=options['num_workers'], shuffle=False)
    # network, loss functions and optimizer
    device: torch.device = torch.device(options['device'])
    net: nn.Module = mymodels.WideResNet(
        depth=options['resnet']['depth'],
        num_classes=options['num_classes'],
        widen_factor=options['resnet']['widen'],
        dropRate=configs['dropout_rate'],
        require_intermediate=False)
    net = net.to(device)
    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: optim.Optimizer = optim.SGD(
        net.parameters(),
        lr=configs['lr_init'],
        weight_decay=configs['weight_decay'],
        momentum=0.9, nesterov=True)
    lr_scheduler: Any = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=configs['lr_stepsize'],
        gamma=configs['lr_gamma'])
    # training
    train_losses: list[float] = list()
    train_accs: list[float] = list()
    val_losses: list[float] = list()
    val_accs: list[float] = list()
    for epoch in range(options['max_epochs']):
        # training
        now_train_loss, now_train_acc = train_loop(
            trainloader, net, device, optimizer, criterion, lr_scheduler, train=True, use_tqdm=options['use_tqdm'])
        train_losses.append(now_train_loss)
        train_accs.append(now_train_acc)
        if api_token is not None:
            run["metrics/train/loss"].append(now_train_loss)
            run["metrics/train/acc"].append(now_train_acc)
        # testing with validation data
        now_val_loss, now_val_acc = train_loop(
            valloader, net, device, optimizer, criterion, train=False, use_tqdm=options['use_tqdm'])
        val_losses.append(now_val_loss)
        val_accs.append(now_val_acc)
        if api_token is not None:
            run["metrics/val/loss"].append(now_val_loss)
            run["metrics/val/acc"].append(now_val_acc)
        # checkpoints
        if now_val_loss==min(val_losses):
            myutils.save_model(
                options['checkpoint_dir'], 'checkpoint_{}.pt'.format(epoch),
                net, optimizer, val_losses, val_accs,
                options['max_epochs'])
        # report
        if api_token is not None:
            session.report({'loss': now_val_loss, 'accuracy': now_val_acc})
    # save the final model
    myutils.save_model(
        options['checkpoint_dir'], 'checkpoint_final.pt',
        net, optimizer, val_losses, val_accs,
        options['max_epochs'])
    return net, optimizer, train_losses, train_accs, val_losses, val_accs

# main function
@hydra.main(version_base=None, config_path='hydra_conf', config_name='neptune_ray_test')
def main(cfg: DictConfig) -> None:
    # initializing ray
    ray.init(num_cpus=16, num_gpus=2)
    # configs and options
    options: dict[str, Any] = OmegaConf.to_container(cfg.options)
    configs: dict[str, Any] = OmegaConf.to_container(cfg.configs)
    ## settting dafault options
    options['dataset']['dir'] = os.getcwd() + '/dataset'
    options['checkpoint_dir'] = os.getcwd() + '/checkpoints'
    ## instanciating every element in configs
    configs['dropout_rate'] = eval(configs['dropout_rate'])
    configs['weight_decay'] = eval(configs['weight_decay'])
    configs['batch_size']   = eval(configs['batch_size'])
    configs['lr_init']      = eval(configs['lr_init'])
    configs['lr_gamma']     = eval(configs['lr_gamma'])
    configs['lr_stepsize']  = eval(configs['lr_stepsize'])
    # scheduler
    scheduler: ASHAScheduler = ASHAScheduler(
        metric='loss', mode='min', max_t=options['max_epochs'],
        grace_period=5, reduction_factor=2)
    # search algorithm
    search_alg: Searcher = OptunaSearch(metric='loss', mode='min')
    search_alg: Searcher = ConcurrencyLimiter(search_alg, max_concurrent=2)
    # Progress reporter
    reporter: CLIReporter = CLIReporter(
        metric_columns=['loss', 'accuracy', 'training_iteration'],
        max_progress_rows=5, max_report_frequency=5)
    # optimization
    ray_tuner: tune.Tuner = tune.Tuner(
        tune.with_resources(
            partial(train, options=options, api_token=cfg.neptune.api_token),
            resources={'cpu': 2, 'gpu': 1},
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=100,
        ),
        param_space=configs,
        run_config=RunConfig(
            name='raytune_with_neptune',
            local_dir='./ray_results',
            progress_reporter=reporter,
        )
    )
    ray_result: tune.ResultGrid = ray_tuner.fit()
    # showing best result
    best_trial: Result = ray_result.get_best_result('loss', 'min', 'last')
    print('Best trial config: {}'.format(best_trial.config))
    print('Best trial final validation loss: {}'.format(best_trial.metrics['loss']))
    print('Best trial final validation accuracy: {}'.format(best_trial.metrics['accuracy']))
    # training with train+val set and best configuration
    trainvalset, trainset, valset, testset = myutils.load_data('./dataset', train_val_ratio=0.9)
    options['datasets'] = {'train': trainvalset, 'val': testset}
    net, optimizer, train_losses, train_accs, test_losses, test_accs = train(best_trial.config, options=options)
    print('Best trial test set accuracy: {}'.format(test_accs[-1]))
    # Save the final model
    myutils.save_model('./model', 'model_final.pt', net, optimizer, test_losses, test_accs, options['max_epochs'])

if __name__=='__main__':
    main()
