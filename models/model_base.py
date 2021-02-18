import torch
import torch.nn as nn
import datetime
import os
import glob2
import torch.nn.functional as F
import math
from models import resnet, resnet18, vgg, mlp, cnn
from losses.softmax import SoftmaxLayer, AngularIsoLoss, IsolateLoss, AngularIsoLoss_v2
from argparse import Namespace
import yaml


class Model_base(nn.Module):
    def __init__(self, args):
        super().__init__()
        # model definition
        if 'resnet' in args.backbone:
            self.backbone = getattr(resnet, args.backbone)(input_channel=1, embed_dim=args.embed_dim)
            # backbone = resnet18.architecture(embed_dim=args.embed_dim)
        elif 'mlp' in args.backbone:
            self.backbone = getattr(mlp, args.backbone)(embed_dim=args.embed_dim)
        elif 'cnn' in args.backbone:
            self.backbone = getattr(cnn, args.backbone)(embed_dim=args.embed_dim)
        elif 'vgg' in args.backbone:
            self.backbone = getattr(vgg, args.backbone)(embed_dim=args.embed_dim)
        else:
            raise ValueError('args.backbone should be resnet or vgg')

        if args.softmax_class == 'oc':
            self.softmax_layer = AngularIsoLoss(embed_dim=args.embed_dim,
                                                r_real=args.r1,
                                                r_fake=args.r2,
                                                alpha=args.softmax_scalar,
                                                norm_v=args.norm_v,
                                                norm_w=args.norm_w,
                                                )
        elif args.softmax_class == 'oc-v2':
            self.softmax_layer = AngularIsoLoss_v2(embed_dim=args.embed_dim,
                                                r_real=args.r1,
                                                r_fake=args.r2,
                                                alpha=args.softmax_scalar,
                                                norm_v=args.norm_v,
                                                norm_w=args.norm_w,
                                                )
        elif args.softmax_class == 'iso':
            self.softmax_layer = IsolateLoss(embed_dim=args.embed_dim)
        elif args.softmax_class == 'mc':
            self.softmax_layer = SoftmaxLayer(embed_dim=args.embed_dim,
                                              num_classes=args.num_classes,
                                              margin=args.softmax_margin,
                                              scalar=args.softmax_scalar,
                                              norm_v=args.norm_v,
                                              norm_w=args.norm_w,
                                              )
        else:
            raise ValueError('args.softmax_class should be oc or mc')

    def forward(self, input, target):
        embeddings = self.backbone(input)
        output, loss = self.softmax_layer(embeddings, target)
        if math.isinf(loss) or math.isnan(loss):
            torch.save(embeddings.detach().cpu(), '/home/fei/err.pt')
            print(loss)
            raise ValueError('Loss error!')
        return output, loss

    @classmethod  # Fixed
    def save(cls, model, path, optimizer, epoch, tr_metric=None, val_metric=None):
        package = cls.serialize(model, optimizer, epoch, tr_metric=tr_metric, val_metric=val_metric)
        torch.save(package, path)

    @classmethod  # Fixed
    def encode_model_identifier(cls,
                                metric_name,
                                metric_value):
        ts = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%s")

        file_identifiers = [metric_name, str(metric_value)]
        model_identifier = "_".join(file_identifiers + [ts])

        return model_identifier

    @classmethod  # Fixed
    def decode_model_identifier(cls,
                                model_identifier):
        identifiers = model_identifier.split("_")
        ts = identifiers[-1].split('.pt')[0]
        [metric_name, metric_value] = identifiers[:-1]
        return metric_name, float(metric_value), ts

    @classmethod  # Fixed
    def get_best_checkpoint_path(cls, model_dir_path):
        best_paths = glob2.glob(model_dir_path + '/best_*')
        if best_paths:
            return best_paths[0]
        else:
            return None

    @classmethod  # Fixed
    def get_current_checkpoint_path(cls, model_dir_path):
        current_paths = glob2.glob(model_dir_path + '/current_*')
        if current_paths:
            return current_paths[0]
        else:
            return None

    @classmethod  # Fixed
    def save_if_best(cls, save_dir, model, optimizer, epoch, tr_metric, val_metric, metric_name, save_every=None):
        '''
        best model is determined by comparing the val_metric
        :param save_dir:
        :param model:
        :param optimizer:
        :param epoch:
        :param tr_metric:
        :param val_metric:
        :param cv_loss_name:
        '''
        if not os.path.exists(save_dir):
            print("Creating non-existing model states directory... {}"
                  "".format(save_dir))
            os.makedirs(save_dir)

        current_path = cls.get_current_checkpoint_path(save_dir)
        models_to_remove = []
        if current_path is not None:
            models_to_remove = [current_path]
        best_path = cls.get_best_checkpoint_path(save_dir)
        file_id = cls.encode_model_identifier(metric_name, val_metric)

        if best_path is not None:
            best_fileid = os.path.basename(best_path)
            _, best_metric_value, _ = cls.decode_model_identifier(
                best_fileid.split('best_')[-1])
        else:
            best_metric_value = 99999999

        if float(val_metric) < float(best_metric_value):
            if best_path is not None:
                models_to_remove.append(best_path)
            save_path = os.path.join(save_dir, 'best_' + file_id + '.pt')
            cls.save(model, save_path, optimizer, epoch,
                     tr_metric=tr_metric, val_metric=val_metric)

        save_path = os.path.join(save_dir, 'current_' + file_id + '.pt')
        cls.save(model, save_path, optimizer, epoch,
                 tr_metric=tr_metric, val_metric=val_metric)

        if save_every:
            if epoch % save_every == 0:
                save_path = os.path.join(save_dir, 'temp{}_'.format(epoch) + file_id + '.pt')
                cls.save(model, save_path, optimizer, epoch,
                         tr_metric=tr_metric, val_metric=val_metric)
        try:
            for model_path in models_to_remove:
                os.remove(model_path)
        except:
            print("Warning: Error in removing {} ...".format(current_path))

    @staticmethod  # Fixed
    def serialize(model, optimizer, epoch, tr_metric=None, val_metric=None):
        package = {}
        package['state_dict'] = model.state_dict()
        package['optim_dict'] = optimizer.state_dict()
        package['epoch'] = epoch
        if tr_metric is not None:
            package['tr_metric'] = tr_metric
            package['val_metric'] = val_metric
        return package

    @classmethod  # Customize the dir_id
    def load_model(cls, models_dir, model_state='best'):
        path = ''
        try:
            path = glob2.glob(models_dir + '/{}_*'.format(model_state))[0]
        except IndexError:
            print('No {} model in {}'.format(model_state, models_dir))

        with open(os.path.join(models_dir, 'logs', 'conf.yml')) as f:
            arg_dic = yaml.safe_load(f)

        model = cls(args=Namespace(**arg_dic))
        print('\nLoad the pre-trained model: {}\n'.format(path))
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model.load_state_dict(package['state_dict'], strict=False)
        return model

    @classmethod
    def load_optimizer(cls, opt=None, models_dir=None, model_state='best'):
        path = ''
        try:
            path = glob2.glob(models_dir + '/{}_*'.format(model_state))[0]
        except IndexError:
            print('No {} model in {}'.format(model_state, models_dir))
        package = torch.load(path, map_location=lambda storage, loc: storage)
        if opt is not None:
            opt.load_state_dict(package['optim_dict'])
        epoch = package['epoch']
        return opt, epoch