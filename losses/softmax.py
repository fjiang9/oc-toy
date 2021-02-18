import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable


from typing import Tuple
from torch import nn, Tensor


def convert_label_to_similarity(feature: Tensor, label: Tensor, sim='dotProduct') -> Tuple[Tensor, Tensor]:
    if sim == 'negL2norm':
        similarity_matrix = torch.zeros(feature.shape[0], feature.shape[0]).cuda()
        for i in range(feature.shape[0]):
            for j in range(feature.shape[0]):
                similarity_matrix[i][j] = 2/(1+torch.exp(torch.dist(feature[i], feature[j], p=2)))
    else:
        normed_feature = F.normalize(feature, p=2, dim=1)
        similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    # find the hardest pair
    # similarity_matrix[]

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    # print(positive_matrix.int().sum(), negative_matrix.int().sum())
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float, beta=1.0) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.beta = beta   # when beta=1 it's circle loss, otherwise, it's ellipse loss
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma * self.beta

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


class SoftmaxLayer(nn.Module):
    def __init__(self, embed_dim=128, num_classes=1251, norm_v=False, norm_w=True,
                 scalar=1.0, margin=0.0):
        super(SoftmaxLayer, self).__init__()
        # if GM is not None:
        #     self.fc = None
        # else:
        #     self.fc = nn.Linear(embedding_dim, int(num_classes), bias=False)
        #     nn.init.kaiming_uniform_(self.fc.weight, 0.25)
        self.num_classes = num_classes
        self.norm_w = norm_w
        self.norm_v = norm_v

        self.W = nn.Parameter(torch.randn(num_classes, embed_dim), requires_grad=True)
        nn.init.kaiming_uniform_(self.W, 0.25)

        self.scalar = scalar
        self.margin = margin
        # self.soft_plus = nn.Softplus()

    def forward(self, x, y):
        if self.norm_w:
            W = F.normalize(self.W, p=2, dim=1)
        else:
            W = self.W
        if self.norm_v:
            x = F.normalize(x, p=2, dim=1)
            
        logits = x @ W.T
        # output_scores = logits.clone()
        output_scores = x @ torch.stack((W[0] - W[1], W[1] - W[0])).T
        if self.margin:
            y_view = y.view(-1, 1)
            if y_view.is_cuda: y_view = y_view.cpu()
            m = torch.zeros(x.shape[0], self.num_classes).scatter_(1, y_view, self.margin)
            if x.is_cuda: m = m.cuda()
            logits = logits - m

        prob = F.log_softmax(self.scalar*logits, dim=1)
        loss = F.nll_loss(prob, y)
        return output_scores, loss


class AngularIsoLoss(nn.Module):
    def __init__(self, embed_dim=2, r_real=0.9, r_fake=0.5, alpha=10.0, norm_v=False, norm_w=True):
        super(AngularIsoLoss, self).__init__()
        self.embed_dim = embed_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.norm_w = norm_w
        self.norm_v = norm_v
        self.W = nn.Parameter(torch.randn(1, self.embed_dim))
        nn.init.kaiming_uniform_(self.W, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, embed_dim).
            labels: ground truth labels with shape (batch_size).
        """
        if self.norm_w:
            w = F.normalize(self.W, p=2, dim=1)
        else:
            w = self.W
        if self.norm_v:
            x = F.normalize(x, p=2, dim=1)

        scores = x @ w.T
        output_scores = scores.clone()
        scores[labels == 0] = self.r_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.r_fake
        loss = self.softplus(torch.logsumexp(self.alpha*scores, dim=0))
        return output_scores, loss


class AngularIsoLoss_v2(nn.Module):
    def __init__(self, embed_dim=2, r_real=0.9, r_fake=0.5, alpha=10.0, norm_v=False, norm_w=True):
        super(AngularIsoLoss_v2, self).__init__()
        self.embed_dim = embed_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.norm_w = norm_w
        self.norm_v = norm_v
        self.W = nn.Parameter(torch.randn(1, self.embed_dim))
        nn.init.kaiming_uniform_(self.W, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, embed_dim).
            labels: ground truth labels with shape (batch_size).
        """
        if self.norm_w:
            w = F.normalize(self.W, p=2, dim=1)
        else:
            w = self.W
        if self.norm_v:
            x = F.normalize(x, p=2, dim=1)

        scores = x @ w.T
        output_scores = scores.clone()
        scores[labels == 0] = self.r_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.r_fake
        loss = self.softplus(torch.logsumexp(self.alpha * scores[labels == 0], dim=0)) + \
               self.softplus(torch.logsumexp(self.alpha * scores[labels == 1], dim=0))
        return output_scores, loss


class IsolateLoss(nn.Module):
    """Isolate loss.
        Reference:
        I. Masi, A. Killekar, R. M. Mascarenhas, S. P. Gurudatt, and W. AbdAlmageed, “Two-branch Recurrent Network for Isolating Deepfakes in Videos,” 2020, [Online]. Available: http://arxiv.org/abs/2008.03412.
        Args:
            feat_dim (int): feature dimension.
        """
    def __init__(self, embed_dim=2, r_real=0.5, r_fake=6.0):
        super(IsolateLoss, self).__init__()
        self.feat_dim = embed_dim
        self.r_real = r_real
        self.r_fake = r_fake

        self.W = nn.Parameter(torch.randn(1, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        loss = F.relu(torch.norm(x[labels==0]-self.W, p=2, dim=1) - self.r_real).mean() \
               + F.relu(self.r_fake - torch.norm(x[labels==1]-self.W, p=2, dim=1)).mean()
        output_scores = -torch.norm(x-self.W, p=2, dim=1, keepdim=True)
        return output_scores, loss


