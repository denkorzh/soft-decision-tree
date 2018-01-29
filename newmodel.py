import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import numpy as np
from scipy.stats import uniform
from torch.autograd import Variable


def get_losses(preds, labels, weights=None):
    from collections import namedtuple
    from sklearn.metrics import (accuracy_score, auc, roc_curve,
                                 average_precision_score,
                                 f1_score, precision_score, recall_score,
                                 log_loss)
    Losses = namedtuple("Losses", ["auc_roc", "acc", "auc_pr", "f1",
                                   "prec", "rec", "log_loss"])
    weights = weights if weights is not None else np.ones_like(labels)

    is_binary = (len(np.unique(labels)) == 2)

    fpr, tpr, thresholds = roc_curve(labels, preds[:, 1])
    auc_roc = auc(fpr, tpr)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(1),
                         sample_weight=weights)
    auc_pr = average_precision_score(labels, preds[:, 1],
                                     sample_weight=weights)
    f1 = f1_score(labels, preds.argmax(1), sample_weight=weights)
    prec = precision_score(labels, preds.argmax(1),
                           sample_weight=weights)
    rec = recall_score(labels, preds.argmax(1), sample_weight=weights)
    eps = np.finfo(np.float32).eps
    log_l = log_loss(labels, np.clip(preds[:, 1], eps, 1 - eps),
                     sample_weight=weights)
    return Losses(auc_roc, acc, auc_pr, f1, prec, rec, log_l)


def save_losses(prefix, losses, writer, step):
    for i, n in enumerate(losses._fields):
        writer.add_scalar("{}/{}".format(prefix, n), losses[i], step)


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
        self.classes_ratio = nn.Parameter(torch.randn(self.args.output_dim))
        if self.args.cuda:
            self.classes_ratio.cuda()
        self.leaf = True
        self.softmax = nn.Softmax()

    def forward(self, *x):
        return self.softmax(self.classes_ratio.view(1, -1))  # 1 x c

    def get_probs(self, x, path_prob):
        """Get the probability of visiting the node and classes distribution"""
        self.path_prob = path_prob
        classes_dist_per_object = self.forward()  # 1 x c
        classes_dist = classes_dist_per_object.expand(x.size()[0], classes_dist_per_object.size()[1])
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
        self.go_right_prob = nn.Sigmoid()(self.forward(x))
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
        assert self.args.mode in {'argmax', 'mean'}, "mode should be 'argmax' or 'mean'"

        self.root = InnerNode(1, self.args)

        assert self.args.l1_mode in {'learnable', 'sampled'}, "l1_mode should be 'learnable' or 'sampled'"
        self.set_alpha()
        if self.args.l1_mode == 'learnable':
            self.alpha = nn.Parameter(torch.FloatTensor([self.args.l1_const]))
            # self.alpha = self.args.l1_const

        self.bn = nn.BatchNorm1d(self.args.input_dim)
        self.train_mode = False

        # self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        # self.scheduler = lrs.ReduceLROnPlateau(self.optimizer, factor=0.9, patience=50, verbose=True, cooldown=50)
        self.scheduler = None

        self.layers = nn.ModuleList()
        self.collect_layers()

    def set_alpha(self, min_val=-10, max_val=0):
        """Set the random value of L1 regularization constant"""
        if self.args.l1_mode == 'sampled':
            self.alpha = uniform(loc=min_val, scale=max_val-min_val).rvs()

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
        visit_root_prob = nn.Parameter(torch.ones(x.size()[0], 1).cuda(), requires_grad=False)
        leaves_probs = self.root.get_probs(x, visit_root_prob)  # [(bs x 1, bs x c), ...]
        leaves_probs = list(zip(*leaves_probs))  # [(bs x 1, bs x 1, ...), (bs x c, bs x c, ...)]
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
        assert ((x is not None) != (paths_probs is not None and distribs is not None),
                'One of x and leaves_probs should be provided')
        if paths_probs is None:
            paths_probs, distribs = self.get_probs(x)

        bs, numleaves, numclasses = distribs.size()
        loss = Variable(torch.zeros(bs, 1).cuda())  # bs x 1

        proba_of_true_label = distribs.gather(dim=2,
                                              index=y.view(-1, 1, 1).expand(bs, numleaves, 1)
                                              ).view(bs, numleaves)  # bs x numleaves
        proba_of_true_label = torch.clamp(proba_of_true_label, min=float(np.finfo(np.float32).eps), max=float(np.inf))
        loss = loss - torch.mul(paths_probs.view(bs, numleaves), torch.log(proba_of_true_label)).sum(dim=1)
        loss = loss.mean()
        return loss

    def get_l1_reg(self):
        """Get L1 regularizator for Linear modules"""
        l1_reg = 0
        for layer in self.layers:
            layer_params = list(layer.parameters())
            l1_reg += layer_params[0].abs().pow(2).sum() + layer_params[1].abs().pow(2)
        if self.args.l1_mode == 'learnable':
            l1_reg = torch.exp(self.alpha) * l1_reg
        elif self.args.l1_mode == 'sampled':
            l1_reg = float(np.exp(self.alpha)) * l1_reg
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
        bs, numleaves, numclasses = distribs.size()
        result = None

        if self.args.mode == 'argmax':
            most_probable_leaf_idx = paths_probs.max(dim=1)[1].view(-1)  # index of most probable leaf (bs)
            expanded_idx = (
                most_probable_leaf_idx
                .view(-1, 1, 1)
                .expand(most_probable_leaf_idx.size()[0], 1, self.args.output_dim)
            )  # bs x 1 x c

            result = distribs.gather(dim=1, index=expanded_idx).view(bs, numclasses)

        elif self.args.mode == 'mean':
            result = torch.mean(paths_probs * distribs, dim=1)  # bs x 1 x c
            result = result.view(bs, numclasses)  # bs x c

        if self.train_mode:
            return result, paths_probs, distribs
        else:
            return result

    def train_epoch(self, data_loader, verbose=True):
        self.train()
        self.train_mode = True
        self.set_alpha()

        for batch_idx, (x, y) in enumerate(data_loader):
            if self.args.cuda:
                x, y = x.cuda(), y.cuda()
            x = x.view(x.size()[0], -1)
            x, y = Variable(x), Variable(y)
            y_est, paths_probs, distribs = self.forward(x)
            loss = (
                    self.get_primary_loss(y, paths_probs=paths_probs, distribs=distribs) +
                    self.get_l1_reg() +
                    self.get_penalty() +
                    0
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.scheduler is None:
                self.optimizer.step()
            else:
                self.scheduler.step(loss.data[0])

            metrics = dict()
            metrics['loss'] = loss.data[0]
            metrics['accuracy'] = np.mean(y.data.eq(y_est.max(dim=1)[1].data).view(-1).cpu().numpy())

            if verbose and not (batch_idx % self.args.log_interval):
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
            x = x.view(x.size()[0], -1)
            x, y = Variable(x), Variable(y)
            y_est = self.forward(x)
            predicted_labels.extend(y_est.max(1)[1].data.view(-1).cpu().numpy())
            true_labels.extend(y.data.view(-1).cpu().numpy())
            if binary_task:
                # predicted_probs.extend(y_est.data.cpu().numpy()[:, 1])
                predicted_probs.extend(y_est.data.cpu().numpy())

        # info = 'Test\t-\tAccuracy: {:.3f}'.format(accuracy_score(true_labels, predicted_labels))
        # if binary_task:
        #     info += '\t-\tAUC ROC: {:.3f}'.format(roc_auc_score(true_labels, predicted_probs))
        #
        # print(info)

        return get_losses(np.array(predicted_probs), true_labels)
