import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal, Laplace

import sys
import os
import time
import numpy as np 

def progress_str(cur_val, max_val, total_point=50):
    p = int(np.ceil(float(cur_val)*total_point/ max_val))
    return '|' + p*'#'+ (total_point - p)*'.'+ '|'

def save_best(loss_best, loss_cur, net, filename, verbose=1):
    if loss_best > loss_cur:
        torch.save(net.state_dict(), filename)
        if verbose:
            print(' --> Save!')
        return True
    else:
        if verbose:
            print()
        return False
def adjust_learning_rate(optimizer, lr_init, epoch, ratio, period):
    lr = lr_init * (ratio ** (epoch // period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_iter(train_loader, net, criterion, optimizer, verbose=1, device=torch.device('cuda')):
    sum_loss = []
    correct = 0
    iter_num = np.ceil(len(train_loader.dataset) / train_loader.batch_size)
    net.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        if device == torch.device('cuda'):
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        if len(labels.shape) > 1:
            labels = torch.argmax(labels, dim=1)

        # forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # GD
        optimizer.zero_grad()             
        loss.backward()                   
        optimizer.step()

        # result
        _, predicted = torch.max(outputs.data, 1) 
        loss_point = loss.float().item()
        correct += (predicted == labels).sum().item()
        sum_loss.append(loss_point)

        # progress bar
        if verbose:
            str0 = progress_str(i+1, iter_num, 50)
            sys.stdout.write("\r%s %.2f%%" % (str0, ((i+1)*100.0)/iter_num))
            sys.stdout.flush()

    train_acc = correct / len(train_loader.dataset)
    train_loss = np.sum(sum_loss) / (i+1)
    return train_acc, train_loss

def my_train(net, train_loader, valid_loader, filename,
             epoch_num, LR, Momentum, patience, 
             device, optimizer, criterion, metric='loss', verbose=1):
    if metric == 'loss':
        metric_best = 1e10
    else:
        metric_best = 0
        
    patience_cnt = 0
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []
    
    for epoch in range(epoch_num):
        print('\r  + {:d}'.format(epoch+1), end='')
        #== LR decay ==
        if epoch < 50:
            adjust_learning_rate(optimizer, LR, epoch, 0.5, 10)
        else:
            adjust_learning_rate(optimizer, LR, epoch, 0.2, 10)
        #== Training ==
        train_acc, train_loss = train_iter(train_loader, net, criterion, optimizer, verbose=verbose, device=device)

        #== Validation ==
        valid_acc, valid_loss = my_eval(net, criterion, valid_loader, verbose=verbose, device=device)
        
        train_acc_list.append(train_acc)
        val_acc_list.append(valid_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(valid_loss)
        
        if verbose:
            print('\r[{:d}] acc: {:.3f}, loss:{:.3f}, valid_acc: {:.3f}, valid_loss:{:.3f}'
            .format(epoch + 1, train_acc, train_loss, valid_acc, valid_loss), end='')
        if metric == 'loss':
            metric_monitor = valid_loss
        else:
            metric_monitor = -valid_acc
        if save_best(metric_best, metric_monitor, net, filename, verbose=verbose):
            patience_cnt = 0
            metric_best = metric_monitor
        else:
            patience_cnt += 1
        if patience_cnt > patience:
            print('\n  + Early Stopping!!!')
            break
    
    train_acc_list = np.array(train_acc_list)
    val_acc_list = np.array(val_acc_list)
    train_loss_list = np.array(train_loss_list)
    val_loss_list = np.array(val_loss_list)
    return(train_acc_list, val_acc_list, train_loss_list, val_loss_list)

def my_eval(net, criterion, test_loader, device=torch.device('cuda'), verbose=1):
    net.eval()
    correct = 0
    sum_loss = 0.0
    iter_num = np.ceil(len(test_loader.dataset) / test_loader.batch_size)
    for i, data_test in enumerate(test_loader):
        images, labels = data_test
        if device == torch.device('cuda'):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
        if len(labels.shape) > 1:
            labels = torch.argmax(labels, dim=1)
        output_test = net(images)
        loss = criterion(output_test, labels)

        _, predicted = torch.max(output_test, 1) 
        correct += (predicted == labels).sum()
        sum_loss += loss.item()
        # progress bar
        if verbose:
            str0 = progress_str(i+1, iter_num, 50)
            sys.stdout.write("\r%s %.2f%%" % (str0, ((i+1)*100.0)/iter_num))
            sys.stdout.flush()

    acc = correct.item() / len(test_loader.dataset)
    loss = sum_loss / (i+1)
    
    return acc, loss

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    storage = os.path.getsize("temp.p")/1e6
    print('Size (MB):', storage)
    os.remove('temp.p')
    return storage

def my_eval_time(net, criterion, test_loader, device=torch.device('cuda'), verbose=1):
    net.eval()
    correct = 0
    sum_loss = 0.0
    elapsed = 0.0
    iter_num = np.ceil(len(test_loader.dataset) / test_loader.batch_size)
    for i, data_test in enumerate(test_loader):
        images, labels = data_test
        if device == torch.device('cuda'):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
        if len(labels.shape) > 1:
            labels = torch.argmax(labels, dim=1)

        start = time.time()
        output_test = net(images)
        end = time.time()
        elapsed = elapsed + (end-start)
        loss = criterion(output_test, labels)

        _, predicted = torch.max(output_test, 1) 
        correct += (predicted == labels).sum()
        sum_loss += loss.item()
        # progress bar
        if verbose:
            str0 = progress_str(i+1, iter_num, 50)
            sys.stdout.write("\r%s %.2f%%" % (str0, ((i+1)*100.0)/iter_num))
            sys.stdout.flush()

    acc = correct.item() / len(test_loader.dataset)
    loss = sum_loss / (i+1)
    
    return acc, loss, elapsed

    #https://github.com/chengyangfu/pytorch-vgg-cifar10