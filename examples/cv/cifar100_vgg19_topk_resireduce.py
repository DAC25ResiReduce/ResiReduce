import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import os
import math
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import numpy as np
import matplotlib.pyplot as plt
import time


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Cifar100 Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--net', default='vgg19',type=str, required=True, help='net type')
parser.add_argument('--model-net', default='vgg19',type=str, help='net type')
parser.add_argument('--train-dir', default=os.path.expanduser('/data/dataset/cv/cifar100/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('/data/dataset/cv/cifar100/validation'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./pytorch_checkpoints/cifar100_vgg19/checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=80,
                    help='number of epochs to train')

parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--gpu', action='store_true', default=True, help='use gpu or not')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

parser.add_argument('--compressor', type=str, default='topk', help='Specify the compressors if density < 1.0')
parser.add_argument('--density', type=float, default=0.01, help='Density for sparsification')

parser.add_argument('--percent', type=float, default=0, help='percent of residual 0')
parser.add_argument('--ptype', type=str, default='/resireduce', help='percent type')
parser.add_argument('--memory', type=str, default='residual', help='Error Feedback')
parser.add_argument('--comm', type=str, default='allgather_fast', help='Communication Library')


y_loss = {}  # loss history
y_loss['train'] = []
y_loss['test'] = []
y_acc = {}
y_acc['train'] = []
y_acc['test'] = []
x_test_epoch_time = []
x_train_epoch_time = []
x_epoch = []


def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)
            if 'cprs' in args.memory:
                optimizer.memory.init_seed()

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            # 保存输出EF信息
            if epoch == 0 and batch_idx == 10 and hvd.rank() == 0 and ('resi' in args.memory or 'cprs' in args.memory or 'tradeoff' in args.memory):
                optimizer.memory.print_ef(epoch, batch_idx)
                # print(f"epoch: {epoch}, idx: {batch_idx}")
                # print(torch.cuda.memory_summary())
            
            if epoch == 0 and batch_idx == 10 and hvd.rank() == 0:
                print(f"epoch: {epoch}, idx: {batch_idx}")
                print(torch.cuda.memory_summary())

            # if batch_idx == 0:
            #     print("batch_idx: ", batch_idx)
            #     print("len(data): ", len(data))
            #     print("data.size(): ", data.size())
            #     # print("data: ", data)
            #     print("target: ", target)
            #     # print("len(target): ", len(target))

            # Split data into sub-batches of size batch_size
            # for i in range(0, len(data), args.batch_size):
            #     data_batch = data[i:i + args.batch_size]
            #     target_batch = target[i:i + args.batch_size]
            #     output = model(data_batch)
            #     train_accuracy.update(accuracy(output, target_batch))
            #     loss = F.cross_entropy(output, target_batch)
            #     train_loss.update(loss)
            #     # Average gradients among sub-batches
            #     loss.div_(math.ceil(float(len(data)) / args.batch_size))
            #     loss.backward()
            
            output = model(data)
            train_accuracy.update(accuracy(output, target))
            loss = F.cross_entropy(output, target)
            train_loss.update(loss)
            loss.backward()
                           
            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)

    # if log_writer:
    #     log_writer.add_scalar('train/loss', train_loss.avg, epoch)
    #     log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)

    y_loss['train'].append(train_loss.avg.item())
    y_acc['train'].append(train_accuracy.avg.item())
    end_time_epoch = time.time()
    x_train_epoch_time.append(end_time_epoch - modified_time)

    # if hvd.rank() == 0:
    #     print('\nTrain set: Average loss: {:.4f}, Train Accuracy: {:.2f}%\n'.format(
    #             train_loss.avg.item(), 100. * train_accuracy.avg.item()))


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')
    val_start_time = time.time()

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    # if log_writer:
    #     log_writer.add_scalar('val/loss', val_loss.avg, epoch)
    #     log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)

    y_loss['test'].append(val_loss.avg.item())
    y_acc['test'].append(val_accuracy.avg.item())
    end_time_epoch = time.time()
    val_time = end_time_epoch - val_start_time
    global modified_time
    modified_time += val_time
    x_test_epoch_time.append(end_time_epoch - modified_time)

    # if hvd.rank() == 0:
    #     print('\nTest set: Average loss: {:.4f}, Test Accuracy: {:.2f}%\n'.format(
    #             val_loss.avg.item(), 100. * val_accuracy.avg.item()))


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    # if epoch < 40:
    #     lr_adj = 1e-1
    # elif epoch < 60:
    #     lr_adj = 1e-2
    # elif epoch < 80:
    #     lr_adj = 1e-3
    # else:
    #     lr_adj = 1e-4
        
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    # for try_epoch in range(args.epochs, 0, -1):
    #     if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
    #         resume_from_epoch = try_epoch
    #         break

    # # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # # checkpoints) to other ranks.
    # resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
    #                                   name='resume_from_epoch').item()

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    # log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'
    
    CIFAR100_TRAIN_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    CIFAR100_TRAIN_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

    # CIFAR100
    train_dataset = \
        datasets.CIFAR100(args.train_dir,
                             train=True,
                             download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=CIFAR100_TRAIN_MEAN,
                                                      std=CIFAR100_TRAIN_STD)
                             ]))
    
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler, **kwargs)
    
    # CIFAR100
    val_dataset = \
        datasets.CIFAR100(args.val_dir,
                             train=False,
                             download=True,
                             transform=transforms.Compose([
                                #  transforms.Resize(256),
                                #  transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=CIFAR100_TRAIN_MEAN,
                                                      std=CIFAR100_TRAIN_STD)
                             ]))    
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                             sampler=val_sampler, **kwargs)

    # Set up standard VGG-19 model.
    model = models.vgg19_bn()
    # model=get_network(args)

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(),
                          lr=(args.base_lr *
                              lr_scaler),
                          momentum=args.momentum, weight_decay=args.wd)



    # allreduce baseline
    if args.compressor == 'none':
        comm_params = {
            'comm_mode':'allreduce',
            'compressor':'none',
            'memory':'none',
            'send_size_aresame':True
        }
    else:
        comm_params = {
            'comm_mode': args.comm,
            'compressor':args.compressor,
            'compress_ratio':args.density,
            'memory':args.memory,
            'percent':args.percent,
            'hrank':hvd.rank(),
            'send_size_aresame':True,
            'model_named_parameters': model.named_parameters()
        }

    # Horovod: wrap optimizer with DistributedOptimizer.
    # 得到一个分布式的SGD优化器

    import resir_lib

    # Horovod: wrap optimizer with DistributedOptimizer.
    # 得到一个分布式的SGD优化器
    optimizer = resir_lib.DistributedOptimizer(
        optimizer, comm_params=comm_params, named_parameters=model.named_parameters())


    start_time = time.time()
    modified_time = start_time
    if hvd.rank() == 0:
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print('start_time_str = ', start_time_str)

    for epoch in range(resume_from_epoch, args.epochs):
        optimizer.init_memory()
        train(epoch)
        # if hvd.rank() == 0 and epoch == 0:
        #     optimizer.output_memory_size()
        validate(epoch)
        
        # 保存最后一个训练模型
        # if epoch==args.epochs-1:
        #     save_checkpoint(epoch)

    if hvd.rank() == 0:
        # torch.cuda.synchronize()
        end_time = time.time()
        end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print('end_time_str = ', end_time_str)
        print('end_time - start_time = ', end_time - start_time)
        training_time = end_time - modified_time
        print('end_time - modified_time = ', training_time)
        print(f"Samples per second: GPU * epochs * iter * batch-size / time = {8 * args.epochs * 196 * 32 / training_time}")
