import sys
import os
import argparse
sys.path.append(os.getcwd())
import logging
import numpy as np
from tqdm import tqdm
from comet_ml import Experiment
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import StepLR
from models.model_base import Model_base
from pprint import pprint
import utils
import json
import torchvision
import torchvision.transforms as transforms
from data import fashionMNIST
from utils import API_KEY, log_metrics, report_losses_mean_and_std
import csv
import yaml
import ast


API_KEY = 'XAM4Urn1JZlSJzvQeNBsKEIYR'
EPS = 1e-8

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='experiments',
                    help='Full path to save best validation model')
# parser.add_argument('--stage', type=int, default=2,
#                     help='stage1: cae pre-training, stage2: embedding net training')
parser.add_argument('--model_dir', type=str, default='tmp')
parser.add_argument('--dataset', type=str, default='fashionmnist')
parser.add_argument('--backbone', type=str, default='resnet18')
parser.add_argument('--embed_dim', type=int, default=2)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--softmax_class', type=str, default='oc')
parser.add_argument('--softmax_margin', type=float, default=0.2)
parser.add_argument('--softmax_scalar', type=float, default=10.0)
parser.add_argument('--norm_w', type=ast.literal_eval, default=True)
parser.add_argument('--norm_v', type=ast.literal_eval, default=True)
parser.add_argument('--r1', type=float, default=0.9)
parser.add_argument('--r2', type=float, default=0.5)
parser.add_argument('--gpu_idx', type=str, nargs="+", default=['3'])



def adjust_learning_rate(optimizer, epoch, initial_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def run_model(model, dataloader, optimizer, train=True, config=None):
    pass


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu_idx)
    # if len(args.gpu_idx):
    #     device = torch.device("cuda" )
    #     print(device)
    # else:
    #     device = torch.device("cpu")


    # dataloader
    input_channel = 1
    if args.dataset == 'fashionmnist':
        args.num_classes = 2

        # args.train_path = '/storage/fei/data/'
        # args.val_path = '/storage/fei/data/'
        # transform = transforms.Compose([transforms.ToTensor(),
        #                                 transforms.Normalize((0.1307,), (0.3081,))])
        #
        # train_set = torchvision.datasets.FashionMNIST(
        #     root=args.train_path,
        #     train=True,
        #     transform=transform
        # )
        # val_set = torchvision.datasets.FashionMNIST(
        #     root=args.val_path,
        #     train=False,
        #     transform=transform
        # )

        from keras.datasets import fashion_mnist
        (trainX, trainy), (testX, testy) = fashion_mnist.load_data()
        # train_set = fashionMNIST(trainX, trainy, real=[5, 7, 9], fake=[0, 1, 2, 3])
        # val_set = fashionMNIST(testX, testy, real=[5, 7, 9], fake=[0, 1, 2, 3, 4, 6, 8])

        real = [3]
        fake_val = [0, 2]
        fake_test = [4, 6]

        train_set = fashionMNIST(trainX, trainy, real=real, fake=fake_val)
        val_set = fashionMNIST(testX, testy, real=real, fake=fake_val)
        test_set = fashionMNIST(testX, testy, real=real, fake=fake_test)
    else:
        raise ValueError('Dataset should be: voxceleb1, imagenet, fashionmnist!')
    #
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)


    model = Model_base(args).cuda()

    experiment = Experiment(API_KEY, project_name='OC-Softmax')
    experiment.log_parameters(vars(args))
    experiment.set_name(args.model_dir)
    numparams = 0
    for f in model.parameters():
        if f.requires_grad:
            numparams += f.numel()
    experiment.log_parameter('Parameters', numparams)
    print('Total number of parameters: {}'.format(numparams))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model.backbone = nn.DataParallel(model.backbone)

    model = model.cuda()

    # Optimizer
    # optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.SGD([{'params': model.backbone.parameters()},
    #                        {'params': model.softmax_layer.parameters()}],
    #                       lr=args.lr, momentum=0.9, nesterov=False)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.6)



    # Save config
    model_path = os.path.join(args.exp_dir, args.model_dir)
    log_path = os.path.join(model_path, 'logs')
    if os.path.exists(log_path):
        res = input("Experiment {} already exists, continue? (y/n)".format(args.model_dir))
        print(res)
        if res == 'n':
            sys.exit()
    os.makedirs(log_path, exist_ok=True)
    conf_path = os.path.join(log_path, 'conf.yml')
    with open(conf_path, 'w') as outfile:
        yaml.safe_dump(vars(args), outfile)

    log_file = '{}/stats.csv'.format(log_path)
    log_content = ['Epoch', 'tr_acc', 'val_acc', 'test_acc', 'val_eer', 'test_eer', 'tr_loss', 'val_loss', 'test_loss']

    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(log_content)

    # Train model
    tr_step = 0
    val_step = 0
    new_lr = args.lr
    halving = False
    best_val_loss = float("-inf")
    val_no_impv = 0

    # training
    iteration = 0
    tr_step = 0
    for epoch in range(args.epochs):
        metric_dic = {}
        for m in log_content[1:]:
            metric_dic[m] = []
        current_lr = adjust_learning_rate(optimizer, tr_step, args.lr)
        # print('Epoch:', epoch,'LR:', optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
        print('Epoch: {}, learning rate: {}'.format(epoch + 1, current_lr))
        # train_utils.val_step(spk_classifier, embedding, val_dataloader,  iteration, val_log_path)



        # Training
        model.train()
        for data in tqdm(train_dataloader, desc='{} Training'.format(args.model_dir)):  # mini-batch
            # one batch of training data
            # input_feature, target = data['input_feature'].to(device), data['target'].to(device)
            input_feature, target = data[0].cuda(), data[1].cuda()

            # gradient accumulates
            optimizer.zero_grad()

            # embedding
            # embeddings = model.backbone(input_feature)
            output, loss = model(input_feature, target)
            metric_dic['tr_loss'].append(loss.detach().cpu())

            # if args.center > 0:
            #     l_c = 0
            #     for i in range(model.embeddings.shape[0]):
            #         l_c = l_c + 0.5 * (model.embeddings[i] - W[:, target[i]]).pow(2).sum()
            #     l_c = l_c / model.embeddings.shape[0]
            #     loss = loss + args.center * l_c
            #     metric_dic['tr_center_loss'].append(l_c.detch().cpu())
            #
            # if args.w_ortho > 0:
            #     W = F.normalize(model.softmax_layer.W, p=2, dim=0)
            #     l_w_reg = (W.T @ W - torch.eye(W.shape[1]).cuda()).norm(p=2)
            #     loss = loss + args.w_ortho * l_w_reg
            #     metric_dic['tr_w_reg'].append(l_w_reg.detach().cpu())

            train_acc = utils.accuracy(output, target)[0]  # Top-1 acc
            metric_dic['tr_acc'].append(train_acc.cpu())

            loss.backward()
            #             torch.nn.utils.clip_grad_norm_(embedding.parameters(), 1.0)
            #             torch.nn.utils.clip_grad_norm_(spk_classifier.parameters(), 1.0)
            optimizer.step()


            if iteration % 100 == 0:
                print('Train loss: {:.2f}, Acc: {:.2f}%'.format(loss.item(), train_acc))

            iteration += 1
        tr_step += 1

        # res_dic['tr_loss']['acc'] += l.tolist()

        # Validation
        if val_dataloader is not None:
            model.eval()
            outputs = []
            targets = []
            with torch.no_grad():
                for data in tqdm(val_dataloader, desc='Validation'):  # mini-batch
                    # input_feature, target = data['input_feature'].to(device), data['target'].to(device)
                    input_feature, target = data[0].cuda(), data[1].cuda()

                    output, loss = model(input_feature, target)

                    # val_acc = utils.accuracy(output, target)[0] # Top-1 acc
                    # metric_dic['val_acc'].append(val_acc.cpu())
                    metric_dic['val_loss'].append(loss.cpu())
                    outputs.append(output)
                    targets.append(target)
            metric_dic['val_acc'] = utils.accuracy(torch.cat(outputs).cpu(), torch.cat(targets).cpu())[0]
            metric_dic['val_acc'] = metric_dic['val_acc'].item()

            eer1, _ = utils.compute_eer(torch.cat(outputs).cpu()[:, 0], torch.cat(targets).cpu())
            eer2, _ = utils.compute_eer(-torch.cat(outputs).cpu()[:, 0], torch.cat(targets).cpu())
            metric_dic['val_eer'] = min(eer1, eer2)

        # Test
        if test_dataloader is not None:
            model.eval()
            outputs = []
            targets = []
            with torch.no_grad():
                for data in tqdm(test_dataloader, desc='Validation'):  # mini-batch
                    # input_feature, target = data['input_feature'].to(device), data['target'].to(device)
                    input_feature, target = data[0].cuda(), data[1].cuda()

                    output, loss = model(input_feature, target)

                    # val_acc = utils.accuracy(output, target)[0] # Top-1 acc
                    # metric_dic['val_acc'].append(val_acc.cpu())
                    metric_dic['test_loss'].append(loss.cpu())
                    outputs.append(output)
                    targets.append(target)
            metric_dic['test_acc'] = utils.accuracy(torch.cat(outputs).cpu(), torch.cat(targets).cpu())[0]
            metric_dic['test_acc'] = metric_dic['test_acc'].item()

            eer1, _ = utils.compute_eer(torch.cat(outputs).cpu()[:, 0], torch.cat(targets).cpu())
            eer2, _ = utils.compute_eer(-torch.cat(outputs).cpu()[:, 0], torch.cat(targets).cpu())
            metric_dic['test_eer']= min(eer1, eer2)


        for metric in metric_dic.keys():
            if isinstance(metric_dic[metric], list):
                metric_dic[metric] = np.mean(metric_dic[metric])
            if metric[:3] == 'tr_':
                with experiment.train():
                    experiment.log_metric(metric[3:], metric_dic[metric], step=tr_step)
            if metric[:4] == 'val_':
                with experiment.validate():
                    experiment.log_metric(metric[4:], metric_dic[metric], step=tr_step)

        pprint(metric_dic)

        # Write logs
        with open(log_file, 'a') as f:
            writer = csv.writer(f)
            write_content = [tr_step] + [metric_dic[m] for m in metric_dic.keys()]
            writer.writerow(write_content)


        Model_base.save_if_best(save_dir=model_path,
                                model=model,
                                optimizer=optimizer,
                                epoch=tr_step,
                                tr_metric=metric_dic['tr_acc'],
                                val_metric=metric_dic['val_eer'],
                                metric_name='eer',
                                save_every=10)


if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler(stream=sys.stdout)], level=logging.INFO,
                        format=' | %(message)s')
    args = parser.parse_args()
    print(args)
    main(args)
