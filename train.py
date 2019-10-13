import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from torch.backends import cudnn
from networks.vgg16_gcn import *
import dataset.data
from tqdm import tqdm
import time
import os
import numpy as np
import pickle
from utils import imgutils
cudnn.enable = True


def validate(model, data_loader):
    print('\nvalidating ... ', flush=True, end='')
    model.eval()
    val_loss = 0
    data_loader = tqdm(data_loader, desc='Validate')
    with torch.no_grad():

        for iter, pack in enumerate(data_loader):
            img = pack[1].cuda()
            target = pack[2].cuda()
            inp = pack[3].cuda()
            x = model(img, inp)
            loss = F.multilabel_soft_margin_loss(x, target)
            val_loss = loss + val_loss

    model.train()
    print('validate loss:', val_loss)
    return


def save_checkpoint(state, filename='checkpoint.pth'):
    print('save model {}'.format(filename))
    torch.save(state, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' GCN Classification')
    parser.add_argument("--batch_size", default=16, type=int, help='mini-batch size')
    parser.add_argument("--network", default="networks.vgg16_gcn", type=str, help='choose the network')
    parser.add_argument("--max_epoches", default=15, type=int,  help='number of total epochs to run')
    parser.add_argument("--lr", default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                        metavar='LR', help='learning rate for pre-trained layers')
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--init_weights", default='vgg_cls.pth', type=str)
    parser.add_argument("--train_list", default="dataset/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="dataset/val.txt", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--inp_name", default="files/voc_glove_word2vec.pkl", type=str)
    parser.add_argument("--dataset_root", default="dataset/VOCdevkit/VOC2012", type=str, help='path to dataset')
    parser.add_argument("--resume", default='', type=str, help='path to latest checkpoint')
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    # print(use_gpu)
    num_classes = 20
    params = {'init_weights': args.init_weights, 'use_gpu': use_gpu}
    model = gcn_vgg16(params=params, num_classes=num_classes, t=0.4, adj_file='files/voc_adj.pkl')
    print(model)
    # VOC12clsDataset里要定义transform
    train_dataset = dataset.data.VOC12clsDataset(args.train_list, path=args.dataset_root,
        inp_name='files/voc_glove_word2vec.pkl',
        transform=transforms.Compose([
                     imgutils.RandomResizeLong(256, 512),
                     transforms.RandomHorizontalFlip(),
                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                     np.asarray,
                     model.normalize,
                     imgutils.RandomCrop(args.crop_size),
                     transforms.ToTensor()]))
    val_dataset = dataset.data.VOC12clsDataset(args.val_list, path=args.dataset_root,
        inp_name='files/voc_glove_word2vec.pkl',
        transform=transforms.Compose([
                        np.asarray,
                        model.normalize,
                        imgutils.CenterCrop(500),
                        transforms.ToTensor()]))

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches
    # 这里先不考虑每层参数的学习率
    if args.resume is not None:   # 加载之前训练好的模型参数
        if os.path.isfile(args.resume):
            print("loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.weight_decay),
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.MultiLabelSoftMarginLoss()
    if use_gpu:
        model.cuda()
        criterion = criterion.cuda()
    #print(model.state_dict())
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    epoch_now = 0
    loss_start_to_end = []
    # print(model.state_dict())
    #torch.save(model.module.state_dict(),  'vg16_gcn.pth')
    for epoch in range(args.max_epoches):
        epoch_now = epoch
        print("epoch[{}/{}]".format(epoch, args.max_epoches))
        all_loss = {}
        all_loss[epoch] = 0
        current_batch_loss = 0
        start = time.time()
        train_data_loader = tqdm(train_data_loader, desc='Training')
        for iter, pack in enumerate(train_data_loader):
            #print(iter)
            iter_50_loss = 0
            img = pack[1].cuda()
            target = pack[2].cuda()
            inp = pack[3].cuda()
            # print(img.shape, inp.shape)
            x = model(img, inp)

            loss = criterion(x, target)
            current_batch_loss = loss
            all_loss[epoch] = all_loss[epoch] + loss
            iter_50_loss = iter_50_loss +loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            #print(iter)
            if (iter+1) % 50 == 0:     # 每50个iter
                during = time.time() - start
                print('Iter:%5d/%5d' % (iter+epoch*661, max_step),
                      'Loss:%.4f' % (iter_50_loss),
                      'Imps:%1f' % ((iter+1) * args.batch_size/during))
                loss_start_to_end.append(iter_50_loss)
        epoch_time = time.time()-start
        print('Loss :current_batch_loss:{}   all_loss:{}    epoch time:{}'
              .format(current_batch_loss, all_loss[epoch], epoch_time))
        validate(model, val_data_loader)   # 每个epoch验证一次
        torch.save(model.module.state_dict(), 'vgg16_gcn' + str(epoch) + '.pth')
    torch.save(model.module.state_dict(), 'vgg16_gcn.pth')









