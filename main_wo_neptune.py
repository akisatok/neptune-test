from tqdm import tqdm
import os
from typing import Any, Callable
import hydra
from omegaconf import OmegaConf, DictConfig

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import myutils
import mymodels

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
        loss_value: float= loss.item()                          
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

def train(configs: dict[str, Any], options: dict[str, Any],
          ) -> tuple[nn.Module, optim.Optimizer, list[float], list[float], list[float], list[float]]:
    # load data
    trainvalset, trainset, valset, testset = myutils.load_data(
        options['dataset']['dir'], train_val_ratio=options['dataset']['train_val_ratio'])
    options['dataset']['train']: Dataset = eval(options['dataset']['train'])
    options['dataset']['val']:   Dataset = eval(options['dataset']['val'])
    options['dataset']['test']:  Dataset = eval(options['dataset']['test'])
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
        print('Epoch {}'.format(epoch+1))
        # training
        now_train_loss, now_train_acc = train_loop(
            trainloader, net, device, optimizer, criterion, lr_scheduler, train=True, use_tqdm=options['use_tqdm'])
        print('Train loss={}, acc={}'.format(now_train_loss, now_train_acc))
        train_losses.append(now_train_loss)
        train_accs.append(now_train_acc)
        # testing with validation data
        now_val_loss, now_val_acc = train_loop(
            valloader, net, device, optimizer, criterion, train=False, use_tqdm=options['use_tqdm'])
        print('Val loss={}, accuracy={}'.format(now_val_loss, now_val_acc))
        val_losses.append(now_val_loss)
        val_accs.append(now_val_acc)
        # checkpoints
        if now_val_loss==min(val_losses):
            myutils.save_model(
                options['checkpoint_dir'], 'checkpoint_{}.pt'.format(epoch),
                net, optimizer, val_losses, val_accs, epoch)
    # save the final model
    myutils.save_model(
        options['checkpoint_dir'], 'checkpoint_final.pt',
        net, optimizer, val_losses, val_accs,
        options['max_epochs']-1)
    return net, optimizer, train_losses, train_accs, val_losses, val_accs

# main function
@hydra.main(version_base=None, config_path='hydra_conf', config_name='neptune_test')
def main(cfg: DictConfig) -> None:
    # configs and options
    options: dict[str, Any] = OmegaConf.to_container(cfg.options)
    configs: dict[str, Any] = OmegaConf.to_container(cfg.configs)
    # settting dafault options
    options['dataset']['dir'] = os.getcwd() + '/dataset'
    options['checkpoint_dir'] = os.getcwd() + '/checkpoints'
    # optimization
    _, _, train_losses, train_accs, test_losses, test_accs = train(configs, options=options)
    # showing the result
    print('Current trial options: {}'.format(options))
    print('Current trial configs: {}'.format(configs))
    print('Current trial final train loss: {}'.format(train_losses[-1]))
    print('Current trial final train accuracy: {}'.format(train_accs[-1]))
    print('Current trial final test loss: {}'.format(test_losses[-1]))
    print('Current trial final test accuracy: {}'.format(test_accs[-1]))

if __name__=='__main__':
    main()
