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
            self.add_module('left_child', InnerNode(self.current_depth + 1, self.args))
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
        self.train_mode = False

        self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.scheduler = lrs.ReduceLROnPlateau(self.optimizer, factor=0.9, patience=50, verbose=True)
        # self.scheduler = None

        self.layers = nn.ModuleList()
        self.collect_layers()

    def collect_layers(self):
        """Collect all Linear layers in the soft tree to one ModuleList"""
        # TODO: it should be a property of an inner node -- to collect subtree Linear layers
        nodes = [self.root]
        while nodes:
            node = nodes.pop(0)
            if not node.leaf:
                self.layers.append(node.fc)
                nodes.append(node._modules['left_child'])
                nodes.append(node._modules['right_child'])

    def get_probs(self, x):
        """Get the probabilities of visiting and respective classes distributions
        of all the leaves in the soft tree"""
        visit_root_prob = torch.ones(x.size()[0], 1)
        leaves_probs = self.root.get_probs(x, visit_root_prob)  # [(bs x 1, bs x c), ...]
        leaves_probs = list(*leaves_probs)  # [(bs x 1, bs x 1, ...), (bs x c, bs x c, ...)]
        paths_probs = torch.stack(leaves_probs[0], dim=1)  # bs x numleaves x 1
        dists = torch.stack(leaves_probs[1], dim=1)  # bs x numleaves x c
        return paths_probs, dists

    def get_primary_loss(self, y, x=None, paths_probs=None, distribs=None):
        """
        Get log-loss weighted with leaves probabilities
        :param y: true class labels (bs x 1)
        :param x: input data (bs x inp)
        :param paths_probs: probabilities of leaves (bs x numleaves x 1)
        :param distribs: distributions of classes in the leaves (bs x numleaves x c)
        :return:
        """
        assert ((x is not None) ^ (paths_probs is not None and distribs is not None),
                'One of x and leaves_probs should be provided')
        if paths_probs is None:
            paths_probs, distribs = self.get_probs(x)

        bs, numleaves, numclasses = distribs.size()
        loss = torch.zeros(bs, 1)  # bs x 1

        proba_of_true_label = distribs.gather(dim=2,
                                              index=y.view(-1, 1, 1).expand(-1, numleaves, 1)
                                              ).view(bs, numleaves)  # bs x numleaves
        loss = loss - torch.mul(paths_probs.view(bs, numleaves), torch.log(proba_of_true_label)).sum(dim=1)
        loss = loss.mean()
        return loss

    def get_l1_reg(self):
        """Get L1 regularizator for Linear modules"""
        l1_reg = 0
        for layer in self.layers:
            layer_params = list(layer.parameters())
            l1_reg += layer_params[0].abs().sum() + layer_params[1].abs()
        l1_reg = self.alpha * l1_reg
        return l1_reg

    def get_penalty(self):
        """Get aggregated soft tree penalty"""
        penalties = self.root.get_penalty()
        penalty = 0.
        for (node_penalty, lmbda) in penalties:
            penalty -= lmbda * 0.5 * (torch.log(node_penalty) + torch.log(1 - node_penalty))
        return penalty

    def forward(self, x):
        x = self.bn(x)
        paths_probs, distribs = self.get_probs(x)  # bs x numleaves x 1,  bs x numleaves x c

        most_probable_leaf_idx = paths_probs.max(dim=1)[1].view(-1)  # index of most probable leaf (bs)
        result = distribs.gather(dim=1,
                                 index=most_probable_leaf_idx.view(-1, 1, 1).expand(-1, 1, self.args.output_dim)
                                 ).view(distribs.size()[[0, 2]])
        if self.train_mode:
            return result, paths_probs, distribs
        else:
            return result

    def train_epoch(self, data_loader, verbose=True):
        self.train()
        self.train_mode = True
        for batch_idx, (x, y) in enumerate(data_loader):
            if self.args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            y_est, paths_probs, distribs = self.forward(x)
            loss = (
                    self.get_primary_loss(y, paths_probs=paths_probs, distribs=distribs) +
                    self.get_l1_reg() +
                    self.get_penalty()
            )
            loss.backward()
            if self.scheduler is None:
                self.optimizer.step()
            else:
                self.scheduler.step(loss.data)

            metrics = dict()
            metrics['loss'] = loss.data
            metrics['accuracy'] = y.data.eq(y_est.max(dim=1)[1].data).view(-1).mean()
            if verbose:
                info = "batch {:d}\t-\tLoss: {:.4f}\t-\tAccuracy: {:.3f}".format(
                    batch_idx,
                    metrics['loss'],
                    metrics['accuracy']
                )
                print(info)

    def print_test_metrics(self, data_loader):
        from sklearn.metrics import accuracy_score, roc_auc_score
        self.eval()
        self.train_mode = False
        predicted_labels = list()
        true_labels = list()
        predicted_probs = list()

        binary_task = (self.args.output_dim == 2)

        for x, y in data_loader:
            if self.args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            y_est = self.forward(x)
            predicted_labels.extend(y_est.max(1)[1].data.view(-1).cpu().numpy())
            true_labels.extend(y.data.view(-1).cpu().numpy())
            if binary_task:
                predicted_probs.extend(y_est.data.cpu().numpy()[:, 1])

        info = 'Test\t-\tAccuracy: {:.3f}'.format(accuracy_score(true_labels, predicted_labels))
        if binary_task:
            info += '\t-\tAUC ROC: {:.3f}'.format(roc_auc_score(true_labels, predicted_probs))

        print(info)
