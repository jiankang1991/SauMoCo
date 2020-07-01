""" 
Deep Embedding Learning for Sen2 L1C dataset based on MoCo
"""

import os
import random
import math
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import shutil

import argparse
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import sys
sys.path.append('../')

from utils.Sen2_model import ResNet18, ResNet50
from utils.metrics import MetricTracker
from utils.Sen2_data_gen import Sen2dataGenNPY, Sen2dataGenLMDB, RandomFlipAndRotate, ToFloatTensor, DataGeneratorSplitting, Normalize
from utils.NCEAverage import MemoryMoCo
from utils.NCECriterion import NCESoftmaxLoss



parser = argparse.ArgumentParser(description='PyTorch SNCA Training for RS')

# parser.add_argument('--IMG_DIR', metavar='DATA_DIR',
#                         help='path to the saved Sen2 L1C train dataset')
parser.add_argument('--LMDB', metavar='DATA_DIR',
                        help='path to the saved Sen2 L1C train dataset')
parser.add_argument('--TESTIMG_DIR', metavar='DATA_DIR',  default='../data',
                        help='path to test eurosat dataset (default: ../data)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--epochs', type=int, default=500, help='epoch number')
parser.add_argument('--neighbor', type=int, default=100, help='neighbor distance')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--num_workers', default=8, type=int, metavar='N',
                        help='num_workers for data loading in pytorch (16 is the default)')
parser.add_argument('--dim', default=128, type=int,
                    metavar='D', help='embedding dimension (default:128)')
parser.add_argument('--nce_t', type=float, default=0.07)
parser.add_argument('--nce_m', type=float, default=0.5)
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')
parser.add_argument('--model', metavar='MODEL',  default='resnet18',
                        help='CNN model (resnet18 or 50)')


args = parser.parse_args()

sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
print('saving file name is ', sv_name)

checkpoint_dir = os.path.join('./', sv_name, 'checkpoints')
logs_dir = os.path.join('./', sv_name, 'logs')

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.isdir(logs_dir):
    os.makedirs(logs_dir)

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (key, str(value)))

def save_checkpoint(state, is_best, name):

    filename = os.path.join(checkpoint_dir, name + '_checkpoint.pth.tar')

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, name + '_model_best.pth.tar'))



def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1-m, p1.detach().data)


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds



def main():
    global args, sv_name, logs_dir, checkpoint_dir

    write_arguments_to_file(args, os.path.join('./', sv_name, sv_name+'_arguments.txt'))

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True
    
    eurosat_norm = {'mean':[1353.73046875, 
                            1117.2020263671875,
                            1041.8876953125,
                            946.5513305664062,
                            1199.1883544921875,
                            2003.0101318359375,
                            2374.01171875,
                            2301.222412109375,
                            732.1828002929688,
                            12.099513053894043,
                            1820.6893310546875,
                            1118.1998291015625,
                            2599.784912109375],
                    'std':[30.343395233154297,
                            66.4549560546875,
                            71.52734375,
                            86.9700698852539,
                            70.47565460205078,
                            81.35286712646484,
                            97.88168334960938,
                            99.96805572509766,
                            27.891748428344727,
                            0.32882159948349,
                            92.60734558105469,
                            87.39993286132812,
                            106.57888793945312]}

    
    
    data_transform = transforms.Compose([
                                        RandomFlipAndRotate(),
                                        ToFloatTensor(),
                                        Normalize(eurosat_norm)
                                        ])

    normalize = transforms.Normalize(mean=eurosat_norm['mean'],
                                    std=eurosat_norm['std'])

    test_data_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize])


    # train_dataGen = Sen2dataGenNPY(
    #                             img_dir=args.IMG_DIR,
    #                             imgTransform=data_transform,
    #                             tile_size_10=64, 
    #                             num_tile=100000, 
    #                             neighborhood=100
    # )

    train_dataGen = Sen2dataGenLMDB(
                                    lmdb_pth=args.LMDB,
                                    imgTransform=data_transform,
                                    tile_size_10=64,
                                    num_tile=100000,
                                    neighborhood=args.neighbor
                                    )

    test_dataGen = DataGeneratorSplitting(
                                    data=args.TESTIMG_DIR, 
                                    dataset='eurosat', 
                                    imgExt='tif',
                                    imgTransform=test_data_transform,
                                    phase='test'
                                    )

    train_data_loader = DataLoader(train_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    test_data_loader = DataLoader(test_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    n_data = len(train_dataGen)

    if args.model == 'resnet18':
        model = ResNet18(dim=args.dim).cuda()
        model_ema = ResNet18(dim=args.dim).cuda()
    else:
        model = ResNet50(dim=args.dim).cuda()
        model_ema = ResNet50(dim=args.dim).cuda()

    moment_update(model, model_ema, 0)

    contrast = MemoryMoCo(args.dim, n_data, n_data, args.nce_t, True).cuda()

    criterion = NCESoftmaxLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=0.9,
                                weight_decay=1e-4, nesterov=True)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    train_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'training'))
    val_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'val'))


    start_epoch = 0
    best_acc = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            model_ema.load_state_dict(checkpoint['model_ema'])

            print("=> loaded successfully '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    for epoch in range(start_epoch, args.epochs):

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        trainMoCo(epoch, train_data_loader, model , model_ema, contrast, criterion, optimizer, train_writer)
        acc = val(test_data_loader, model, epoch, val_writer)

        is_best_acc = acc > best_acc
        best_acc = max(best_acc, acc)

        save_checkpoint({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'model': model.state_dict(),
            'model_ema':model_ema.state_dict(),
            'contrast':contrast.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best_acc, sv_name)

        scheduler.step()

def trainMoCo(epoch, trainloader, model, model_ema, contrast, criterion, optimizer, train_writer):

    loss_meter = MetricTracker()

    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()
    model_ema.apply(set_bn_train)

    for idx, data in enumerate(tqdm(trainloader, desc="training")):

        imgs = data['anchor'].to(torch.device("cuda"))
        aug_imgs = data['neighbor'].to(torch.device("cuda"))
        index = data['idx'].to(torch.device("cuda"))

        bsz = imgs.size(0)

        shuffle_ids, reverse_ids = get_shuffle_ids(bsz)

        feat_q = model(imgs)
        with torch.no_grad():
            aug_imgs = aug_imgs[shuffle_ids]
            feat_k = model_ema(aug_imgs)
            feat_k = feat_k[reverse_ids]

        out = contrast(feat_q, feat_k)

        loss = criterion(out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), bsz)
        moment_update(model, model_ema, args.alpha)

    info = {
        "Loss": loss_meter.avg,

    }
    for tag, value in info.items():
        train_writer.add_scalar(tag, value, epoch)

    print('Train Loss: {:.6f} '.format(
            loss_meter.avg
            ))


def val(valloader, model, epoch, val_writer):
    """ 
    validation on the NAIP test dataset 
    https://github.com/ermongroup/tile2vec/blob/master/examples/Example%203%20-%20Tile2Vec%20features%20for%20CDL%20classification.ipynb
    """
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(valloader, desc="extracting data embeddings")):

            imgs = data['img'].to(torch.device("cuda"))
            label_batch = data['label'].to(torch.device("cpu"))

            f = model(imgs)

            features += list(f.cpu().numpy().astype(np.float32))
            labels += list(np.squeeze(label_batch.numpy()).astype(np.float32))

    features = np.asarray(features)
    labels = np.asarray(labels)


    n_trials = 100
    accs = np.zeros((n_trials,))

    for i in range(n_trials):
        X_tr, X_te, y_tr, y_te = train_test_split(features, labels, test_size=0.2)
        rf = RandomForestClassifier(n_jobs=-1)
        rf.fit(X_tr, y_tr)
        accs[i] = rf.score(X_te, y_te)

    val_writer.add_scalar('RF-Acc', accs.mean(), epoch)
    val_writer.add_scalar('RF-std', accs.std(), epoch)

    print('Validation RF-Acc: {:.6f} '.format(
            accs.mean(),
            # hammingBallRadiusPrec.val,
            ))
    
    return accs.mean()

if __name__ == '__main__':
    main()



