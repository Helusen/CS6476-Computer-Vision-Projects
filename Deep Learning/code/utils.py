import sys
import os
import os.path as osp
import shutil
import time
import random
import numpy as np
from visdom import Visdom

import torch
import torch.utils.data
from torch.autograd import Variable
from IPython.core.debugger import set_trace

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

def set_seed(seed, use_GPU=False):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  if use_GPU:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True      

def print_input_size_hook(self, input, output):
  print('Input size to classifier is', input[0].size())

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res

class Trainer(object):
  def __init__(self, train_dataset, val_dataset, model, loss_fn, optimizer,
      lr_scheduler, params):
    """
    General purpose training script
    :param train_dataset: PyTorch dataset that loads training images
    :param val_dataset: PyTorch dataset that loads testing / validation images
    :param model: Network model
    :param optimizer: PyTorch optimizer object
    :param lr_scheduler: PyTorch learning rate scheduler object
    :param loss_fn: loss function
    :param params: dictionary containing parameters for the training process
    It can contain the following fields (fields with no default value mentioned
    are mandatory):
      n_epochs: number of epochs of training
      batch_size: batch size for one iteration
      do_val: perform validation? (default: True)
      shuffle: shuffle training data? (default: True)
      num_workers: number of CPU threads for loading data (default: 4)
      val_freq: frequency of validation (in number of epochs) (default: 1)
      print_freq: progress printing frequency (in number of iterations
        (default: 20)
      experiment: name of the experiment, used to create logs and checkpoints
      checkpoint_file: Name of file with saved weights. Loaded at before
        start of training if provided (default: None)
      resume_optim: whether to resume optimization from loaded weights
        (default: True)
    """
    self.model = model
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.lr_scheduler = lr_scheduler
    self.best_prec1 = -float('inf')

    # parse params with default values
    self.config = {
      'n_epochs': params['n_epochs'],
      'batch_size': params['batch_size'],
      'do_val': params.get('do_val', True),
      'shuffle': params.get('shuffle', True),
      'num_workers': params.get('num_workers', 4),
      'val_freq': params.get('val_freq', 1),
      'print_freq': params.get('print_freq', 100),
      'experiment': params['experiment'],
      'checkpoint_file': params.get('checkpoint_file'),
      'resume_optim': params.get('resume_optim', True)
    }

    self.logdir = osp.join(os.getcwd(), 'logs', self.config['experiment'])
    if not osp.isdir(self.logdir):
      os.makedirs(self.logdir)

    # visdom plots
    self.vis_env = self.config['experiment']
    self.loss_win = 'loss_win'
    self.vis = Visdom()
    self.vis.line(X=np.zeros((1,2)), Y=np.zeros((1,2)), win=self.loss_win,
      opts={'legend': ['train_loss', 'val_loss'], 'xlabel': 'epochs',
            'ylabel': 'loss'}, env=self.vis_env)
    self.lr_win = 'lr_win'
    self.vis.line(X=np.zeros(1), Y=np.zeros(1), win=self.lr_win,
      opts={'legend': ['learning_rate'], 'xlabel': 'epochs',
            'ylabel': 'log(lr)'}, env=self.vis_env)
    self.top1_win = 'top1_win'
    self.vis.line(X=np.zeros((1,2)), Y=np.zeros((1,2)), win=self.top1_win,
      opts={'legend': ['train_top1_prec', 'val_top1_prec'], 'xlabel': 'epochs',
            'ylabel': 'top1_prec (%)'}, env=self.vis_env)
    self.top5_win = 'top5_win'
    self.vis.line(X=np.zeros((1,2)), Y=np.zeros((1,2)), win=self.top5_win,
      opts={'legend': ['train_top5_prec', 'val_top5_prec'], 'xlabel': 'epochs',
            'ylabel': 'top5_prec (%)'}, env=self.vis_env)

    # log all the command line options
    print('---------------------------------------')
    print('Experiment: {:s}'.format(self.config['experiment']))
    for k, v in self.config.items():
      print('{:s}: {:s}'.format(k, str(v)))
    print('---------------------------------------')

    self.start_epoch = int(0)
    checkpoint_file = self.config['checkpoint_file']
    if checkpoint_file:
      if osp.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_prec1 = checkpoint['best_prec1']
        if self.config['resume_optim']:
          self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
          self.start_epoch = checkpoint['epoch']
        print('Loaded checkpoint {:s} epoch {:d}'.format(checkpoint_file,
          checkpoint['epoch']))

    self.train_loader = torch.utils.data.DataLoader(train_dataset,
      batch_size=self.config['batch_size'], shuffle=self.config['shuffle'],
      num_workers=self.config['num_workers'])
    if self.config['do_val']:
      self.val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=self.config['batch_size'], shuffle=False,
        num_workers=self.config['num_workers'])
    else:
      self.val_loader = None

  def save_checkpoint(self, epoch, is_best):
    filename = osp.join(self.logdir, 'checkpoint.pth.tar')
    checkpoint_dict =\
      {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
       'optim_state_dict': self.optimizer.state_dict(),
       'best_prec1': self.best_prec1}
    torch.save(checkpoint_dict, filename)
    if is_best:
      shutil.copyfile(filename, osp.join(self.logdir, 'best_model.pth.tar'))

  def step_func(self, train):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if train:
      self.model.train()
      status = 'train'
      loader = self.train_loader
    else:
      self.model.eval()
      status = 'val'
      loader = self.val_loader

    end = time.time()

    for batch_idx, (data, target) in enumerate(loader):
      data_time.update(time.time() - end)

      kwargs = dict(target=target, loss_fn=self.loss_fn,
        optim=self.optimizer, train=train)
      loss, output = step_feedfwd(data, self.model, **kwargs)

      # measure accuracy and calculate loss
      prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
      losses.update(loss, data.size(0))
      top1.update(prec1[0], data.size(0))
      top5.update(prec5[0], data.size(0))

      # measure batch time
      batch_time.update(time.time() - end)
      end = time.time()

      if batch_idx % self.config['print_freq'] == 0:
        print('{:s} {:s}: batch {:d}/{:d}, loss {:4.3f}, top-1 accuracy {:4.3f},'
              ' top-5 accuracy {:4.3f}'.format(status, self.config['experiment'],
          batch_idx, len(loader)-1, loss, prec1[0], prec5[0]))

    print('{:s} {:s}: loss {:f}'.format(status, self.config['experiment'],
      losses.avg))

    return losses.avg, top1.avg, top5.avg

  def train_val(self):
    for epoch in range(self.start_epoch, self.config['n_epochs']):
      print('{:s} Epoch {:d} / {:d}'.format(self.config['experiment'], epoch,
        self.config['n_epochs']))

      # ADJUST LR
      self.lr_scheduler.step()
      lr = self.lr_scheduler.get_lr()[0]
      self.vis.line(X=np.asarray([epoch]), Y=np.asarray([np.log10(lr)]),
        win=self.lr_win, name='learning_rate', update='append', env=self.vis_env)

      # TRAIN
      loss, top1_prec, top5_prec = self.step_func(train=True)
      self.vis.line(X=np.asarray([epoch]), Y=np.asarray([loss]),
        win=self.loss_win, name='train_loss', update='append', env=self.vis_env)
      self.vis.line(X=np.asarray([epoch]), Y=np.asarray([top1_prec]),
        win=self.top1_win, name='train_top1_prec', update='append', env=self.vis_env)
      self.vis.line(X=np.asarray([epoch]), Y=np.asarray([top5_prec]),
        win=self.top5_win, name='train_top5_prec', update='append', env=self.vis_env)
      self.vis.save(envs=[self.vis_env])

      # VALIDATION
      if self.config['do_val'] and ((epoch % self.config['val_freq'] == 0) or
                                    (epoch == self.config['n_epochs']-1)):
        loss, top1_prec, top5_prec = self.step_func(train=False)
        self.vis.line(X=np.asarray([epoch]), Y=np.asarray([loss]),
          win=self.loss_win, name='val_loss', update='append', env=self.vis_env)
        self.vis.line(X=np.asarray([epoch]), Y=np.asarray([top1_prec]),
          win=self.top1_win, name='val_top1_prec', update='append', env=self.vis_env)
        self.vis.line(X=np.asarray([epoch]), Y=np.asarray([top5_prec]),
          win=self.top5_win, name='val_top5_prec', update='append', env=self.vis_env)
        self.vis.save(envs=[self.vis_env])

      # SAVE CHECKPOINT
      is_best = top1_prec > self.best_prec1
      self.best_prec1 = max(self.best_prec1, top1_prec)
      self.save_checkpoint(epoch, is_best)
      print('Checkpoint saved')
      if is_best:
        print('BEST TOP1 ACCURACY SO FAR')

    return self.best_prec1

def step_feedfwd(data, model, target=None, loss_fn=None, optim=None,
    train=True):
  """
  training/validation step for a feedforward NN
  :param data:
  :param target:
  :param model:
  :param loss_fn:
  :param optim:
  :param train: training / val stage
  :return:
  """
  if train:
    assert loss_fn is not None

  with torch.no_grad():
    data_var = Variable(data, requires_grad=train)
  output = model(data_var)

  if loss_fn is not None:
    with torch.no_grad():
      target_var = Variable(target, requires_grad=False)
    loss = loss_fn(output, target_var)
    if train:
      # SGD step
      optim.zero_grad()
      loss.backward()
      optim.step()

    return loss.item(), output
  else:
    return 0, output

def get_mean_std(data_path, input_size, rgb):
  tform = []
  tform.append(transforms.Resize(size=input_size))
  if not rgb:
    tform.append(transforms.Grayscale())
  tform.append(transforms.ToTensor())
  tform = transforms.Compose(tform)
  dset = datasets.ImageFolder(root=data_path, transform=tform)
  train_loader = DataLoader(dataset=dset, batch_size=50)
  scaler = StandardScaler(with_mean=True, with_std=True)
  print('Computing pixel mean and stdev...')
  for idx, (data, labels) in enumerate(train_loader):
    if idx % 20 == 0:
      print("Batch {:d} / {:d}".format(idx, len(train_loader)))
    data = data.numpy()
    n_channels = data.shape[1]
    # reshape into [n_pixels x 3]
    data = data.transpose((0, 2, 3, 1)).reshape((-1, n_channels))
    # pass batch to incremental mean and stdev calculator
    scaler.partial_fit(data)
  print('Done, mean = ')
  pixel_mean = scaler.mean_
  pixel_std  = scaler.scale_
  print(pixel_mean)
  print('std = ')
  print(pixel_std)
  return pixel_mean, pixel_std 
