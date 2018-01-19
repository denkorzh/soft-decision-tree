import os

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score


class Node(nn.Module):
    def __init__(self):
        super(Node, self).__init__()
        self.args = None
        self.path_prob = None  # probability of node visiting
        self.go_right_prob = None  # former self.prob
        self.fc = None
        self.softmax = None
        self.leaf = False
        self.lmbda = None
        self.classes_ratio = None
        self.subtree_probs = None
        self.current_depth = None

    def forward(self, x):
        pass


class LeafNode(Node):
    def __init__(self, args):
        super(LeafNode, self).__init__()
        self.args = args
        self.classes_dist = nn.Parameter(torch.randn(self.args.output_dim))
        if self.args.cuda:
            self.classes_dist.cuda()
        self.leaf = True
        self.softmax = nn.Softmax()

    def forward(self, *x):
        return self.softmax(self.classes_ratio.view(1, -1))  # bs x c

    def get_probs(self, x, path_prob):
        """Get the probability of visiting the node and classes distribution"""
        self.path_prob = path_prob
        classes_dist = self.forward()
        return [(path_prob, classes_dist)]  # path_prob: bs x 1, classes_dist: bs x c


class InnerNode(Node):
    def __init__(self, cur_depth, args):
        super(InnerNode, self).__init__()
        self.args = args
        self.current_depth = cur_depth
        self.fc = nn.Linear(self.args.input_dim, 1)
        self.lmbda = self.args.lmbda * 2 ** (- self.current_depth)
        self.penalties = list()
        self.build_children()

    def build_children(self):
        if self.current_depth < self.args.max_depth:
            self.add_module('left_child', InnerNode(self.current_depth+1, self.args))
            self.add_module('right_child', InnerNode(self.current_depth + 1, self.args))
        else:
            self.add_module('left_child', LeafNode(self.args))
            self.add_module('right_child', LeafNode(self.args))

    def forward(self, x):
        return self.fc(x)

    def get_probs(self, x, path_prob):
        """Get the probabilities of visiting and respective classes distributions
        of all the leaves in the respective subtree"""
        self.path_prob = path_prob
        self.go_right_prob = self.forward(x)
        subtree_leaves_probs = list()
        subtree_leaves_probs.extend(self
                                    ._modules['left_child']
                                    .get_probs(x, self.path_prob * (1 - self.go_right_prob))
                                    )
        subtree_leaves_probs.extend(self
                                    ._modules['right_child']
                                    .get_probs(x, self.path_prob * self.go_right_prob)
                                    )
        return subtree_leaves_probs

    def get_penalty(self):
        """Get penalties for every inner node in the subtree"""
        penalty = [(torch.sum(self.go_right_prob * self.path_prob) / torch.sum(self.path_prob),
                   self.lmbda)]
        if self.current_depth < self.args.max_depth - 1:
            penalty.extend(self._modules['left_child'].get_penalty())
            penalty.extend(self._modules['right_child'].get_penalty())
        return penalty


class SoftDecisionTree(nn.Module):
    def __init__(self, args):
        super(SoftDecisionTree, self).__init__()
        self.args = args
        self.root = InnerNode(1, self.args)
        self.alpha = nn.Parameter(torch.FloatTensor([0.01]))
        self.bn = nn.BatchNorm1d(self.args.input_dim)

        self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.scheduler = lrs.ReduceLROnPlateau(self.optimizer, factor=0.9, patience=50, verbose=True)
        # self.scheduler = None

    def get_probs(self, x):
        """Get the probabilities of visiting and respective classes distributions
        of all the leaves in the soft tree"""
        visit_root_prob = torch.ones(x.size()[0], 1)
        return self.root.get_probs(x, visit_root_prob)  # [(bs x 1, bs x c), ...]

    def get_primary_loss(self, y, x=None, leaves_probs=None):
        """
        Get log-loss weighted with leaves probabilities
        :param y: true class labels (bs x 1)
        :param x: input data (bs x inp)
        :param leaves_probs: [(leaf probability, classes distribution), ...] (bs x 1, bs x c) each one
        :return:
        """
        assert (x is not None) ^ (leaves_probs is not None), 'One of x and leaves_probs should be provided'
        if leaves_probs is None:
            leaves_probs = self.get_probs(x)
        size = leaves_probs[0][1].size()[0]  # get bs
        loss = torch.zeros(size, 1)  # bs x 1

        for path_prob, classes_dist in leaves_probs:
            proba_of_true_label = classes_dist.gather(1, y.view(-1, 1))  # bs x 1
            loss = loss - path_prob * torch.log(proba_of_true_label)

        loss = loss.mean()
        return loss
