import logging
import os
import re
from glob import glob
import json
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import pandas as pd
from sklearn.metrics import roc_curve, auc

API_KEY = 'XAM4Urn1JZlSJzvQeNBsKEIYR'

def compute_eer(y_score, y_true):
    fpr, tpr, threshold = roc_curve(y_true.view(-1), y_score.view(-1), pos_label=1)
    fnr = 1 - tpr
    # print(fpr, tpr)
    eer_threshold = threshold[np.argmin(np.abs(fnr - fpr))]
    EER = fpr[np.argmin(np.abs(fnr - fpr))]
    return EER*100, eer_threshold


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def txt_to_csv(txt_path, data_folder, csv_path="metadata", data_suffix='.pkl'):
    '''
    Parse the Voxceleb1 iden_split.txt file to train, val, test csv meta files
    :param txt_path: path to the iden_split.txt file
    :param csv_path:
    :return:
    '''
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    data_path = {'train': [], 'val': [], 'test': []}
    spk_ids = {'train': [], 'val': [], 'test': []}
    frame_num = {'train': [], 'val': [], 'test': []}
    with open(txt_path, 'r') as f:
        data = f.readlines()
        for line in data:
            content = line.split(' ')
            id_temp = content[1].split('/')[0]
            path_temp = os.path.join(data_folder, content[1].split('.')[0].replace('/', '-') + data_suffix)
            tf_data = pickle2array(path_temp)
            if content[0] == '1':
                spk_ids['train'].append(id_temp)
                data_path['train'].append(path_temp)
                frame_num['train'].append(tf_data.shape[-1])
            if content[0] == '2':
                spk_ids['val'].append(id_temp)
                data_path['val'].append(path_temp)
                frame_num['val'].append(tf_data.shape[-1])
            if content[0] == '3':
                spk_ids['test'].append(id_temp)
                data_path['test'].append(path_temp)
                frame_num['test'].append(tf_data.shape[-1])
    for set in data_path.keys():
        dataframe = pd.DataFrame({'data_path': data_path[set], 'spk_id': spk_ids[set], 'frame_num': frame_num[set]})
        dataframe.to_csv(os.path.join(csv_path, "Vox1_{}.csv".format(set)), index=False)

def log_metrics(metrics_dict, dirpath, step):
    """Logs the accumulative individual results from a dictionary of metrics

    Args:
        metrics_dict: Python Dict with the following structure:
                     res_dic[loss_name] = {'mean': 0., 'std': 0., 'acc': []}
        dirpath:  An absolute path for saving the metrics into
        step:     The step/epoch index
    """

    for metric_name, metric_data in metrics_dict.items():
        this_metric_folder = os.path.join(dirpath, metric_name)
        if not os.path.exists(this_metric_folder):
            print("Creating non-existing metric log directory... {}"
                  "".format(this_metric_folder))
            os.makedirs(this_metric_folder)

        values = metric_data['acc']
        values = np.array(values)
        if 'tr' in metric_name:
            filename = 'epoch_{}'.format(step)
        else:
            filename = 'epoch_{}'.format(step)
        np.save(os.path.join(this_metric_folder, filename), values)

def report_losses_mean_and_std(losses_dict, experiment, step):
    """Wrapper for cometml loss report functionality.

    Reports the mean and the std of each loss by inferring the train and the
    val string and it assigns it accordingly.

    Args:
        losses_dict: Python Dict with the following structure:
                     res_dic[loss_name] = {'mean': 0., 'std': 0., 'acc': []}
        experiment:  A cometml experiment object
        step:     The step/epoch index

    Returns:
        The updated losses_dict with the current mean and std
    """

    for l_name in losses_dict:
        values = losses_dict[l_name]['acc']
        mean_metric = np.mean(values)
        std_metric = np.std(values)

        if 'tr' in l_name:
            actual_name = l_name.replace('tr_', '')
            with experiment.train():
                experiment.log_metric(actual_name + '_mean',
                                      mean_metric,
                                      step=step)
                experiment.log_metric(actual_name + '_std',
                                      std_metric,
                                      step=step)

        elif 'val' in l_name:
            actual_name = l_name.replace('val_', '')
            with experiment.validate():
                experiment.log_metric(actual_name + '_mean',
                                      mean_metric,
                                      step=step)
                experiment.log_metric(actual_name + '_std',
                                      std_metric,
                                      step=step)
        else:
            raise ValueError("tr or val must be in metric name <{}>."
                             "".format(l_name))

        losses_dict[l_name]['mean'] = mean_metric
        losses_dict[l_name]['std'] = std_metric

    return losses_dict



if __name__ == "__main__":
    txt_to_csv('/storage/fei/data/VoxCeleb1/iden_split.txt',
               '/storageNVME/fei/data/speech/vox_features',
               '/home/fei/projects/gm-softmax/metadata')
