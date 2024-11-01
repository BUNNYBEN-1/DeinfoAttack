from abc import ABC, ABCMeta, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_sparse import SparseTensor
import utils as _utils

from utils import gen_pseudo_label

class Attack(metaclass=ABCMeta):
    r"""

    Description
    -----------
    Abstract class for graph adversarial attack.

    """
    @abstractmethod
    def attack(self, model, adj, features, **kwargs):
        r"""

        Parameters
        ----------
        model : torch.nn.module
            Model implemented based on ``torch.nn.module``.
        adj : scipy.sparse.csr.csr_matrix
            Adjacency matrix in form of ``N * N`` sparse matrix.
        features : torch.FloatTensor
            Features in form of ``N * D`` torch float tensor.
        kwargs :
            Keyword-only arguments.

        """


class ModificationAttack(Attack):
    r"""

    Description
    -----------
    Abstract class for graph modification attack.

    """
    @abstractmethod
    def attack(self, **kwargs):
        """

        Parameters
        ----------
        kwargs :
            Keyword-only arguments.

        """

    @abstractmethod
    def modification(self, **kwargs):
        """

        Parameters
        ----------
        kwargs :
            Keyword-only arguments.

        """


class InjectionAttack(Attack):
    r"""

    Description
    -----------
    Abstract class for graph injection attack.

    """
    @abstractmethod
    def attack(self, **kwargs):
        """

        Parameters
        ----------
        kwargs :
            Keyword-only arguments.

        """

    @abstractmethod
    def injection(self, **kwargs):
        """

        Parameters
        ----------
        kwargs :
            Keyword-only arguments.

        """

    @abstractmethod
    def update_features(self, **kwargs):
        """

        Parameters
        ----------
        kwargs :
            Keyword-only arguments.

        """


class EarlyStop(object):
    r"""

    Description
    -----------
    Strategy to early stop attack process.

    """
    def __init__(self, patience=1000, epsilon=1e-4):
        r"""

        Parameters
        ----------
        patience : int, optional
            Number of epoch to wait if no further improvement. Default: ``1000``.
        epsilon : float, optional
            Tolerance range of improvement. Default: ``1e-4``.

        """
        self.patience = patience
        self.epsilon = epsilon
        self.min_score = None
        self.stop = False
        self.count = 0

    def __call__(self, score):
        r"""

        Parameters
        ----------
        score : float
            Value of attack acore.

        """
        if self.min_score is None:
            self.min_score = score
        elif self.min_score - score > 0:
            self.count = 0
            self.min_score = score
        elif self.min_score - score < self.epsilon:
            self.count += 1
            if self.count > self.patience:
                self.stop = True

class AttackABC(ABC):
    """
    attack class的基类
    构造函数inputs：
        - 超参数attack_config，PRBCD、GreedyRBCD、PGA等
        - 数据输入pyg_data
        - 代理模型/攻击对象model，如果直接把攻击对象传进来就是白盒攻击；只是利用model生成adj_adversary，然后攻击其他模型，就是灰盒攻击
        - GPU/CPU device
        - 日志logger
    """
    def __init__(self,
                 attack_config, pyg_data,
                 model, device):
        self.device = device
        self.attack_config = attack_config

        self.loss_type = attack_config['loss_type']
        # self.attacked_model = deepcopy(model).to(self.device)
        self.attacked_model = model  # 注意这里直接引用, 源代码实现是深拷贝
        self.attacked_model.eval()

        self.pyg_data = deepcopy(pyg_data)
        # if self.__class__.__name__ not in ['DICE', 'Random']:  # 给这两个方法真实标签
        #     pseudo_label = gen_pseudo_label(self.attacked_model, self.pyg_data.y, self.pyg_data.test_mask)
        #     self.pyg_data.y = pseudo_label
        pseudo_label = gen_pseudo_label(self.attacked_model, self.pyg_data.y, self.pyg_data.test_mask)
        self.pyg_data.y = pseudo_label
        self.pyg_data = self.pyg_data.to(self.device)


        for p in self.attacked_model.parameters():
            p.requires_grad = False
        self.eval_model = self.attacked_model

        self.attr_adversary = self.pyg_data.x
        self.adj_adversary = self.pyg_data.adj_t


    @abstractmethod
    def _attack(self, n_perturbations):
        pass

    def attack(self, n_perturbations, **kwargs):
        if n_perturbations > 0:
            return self._attack(n_perturbations, **kwargs)
        else:
            self.attr_adversary = self.pyg_data.x
            self.adj_adversary = self.pyg_data.adj_t

    def get_perturbations(self):
        adj_adversary, attr_adversary = self.adj_adversary, self.attr_adversary
        if isinstance(self.adj_adversary, torch.Tensor):
            adj_adversary = SparseTensor.from_dense(self.adj_adversary)
        if isinstance(self.attr_adversary, SparseTensor):
            attr_adversary = self.attr_adversary.to_dense()

        return adj_adversary, attr_adversary

    def calculate_loss(self, logits, labels):
        if self.loss_type == 'CW':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -torch.clamp(margin, min=0).mean()
        elif self.loss_type == 'LCW':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -F.leaky_relu(margin, negative_slope=0.1).mean()
        elif self.loss_type == 'tanhMargin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = torch.tanh(-margin).mean()
        elif self.loss_type == 'Margin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -margin.mean()
        elif self.loss_type.startswith('tanhMarginCW-'):
            alpha = float(self.loss_type.split('-')[-1])
            assert alpha >= 0, f'Alpha {alpha} must be greater or equal 0'
            assert alpha <= 1, f'Alpha {alpha} must be less or equal 1'
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = (alpha * torch.tanh(-margin) - (1 - alpha) * torch.clamp(margin, min=0)).mean()
        elif self.loss_type.startswith('tanhMarginMCE-'):
            alpha = float(self.loss_type.split('-')[-1])
            assert alpha >= 0, f'Alpha {alpha} must be greater or equal 0'
            assert alpha <= 1, f'Alpha {alpha} must be less or equal 1'

            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )

            not_flipped = logits.argmax(-1) == labels

            loss = alpha * torch.tanh(-margin).mean() + (1 - alpha) * \
                F.cross_entropy(logits[not_flipped], labels[not_flipped])
        elif self.loss_type == 'eluMargin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -F.elu(margin).mean()
        elif self.loss_type == 'MCE':
            not_flipped = logits.argmax(-1) == labels
            loss = F.cross_entropy(logits[not_flipped], labels[not_flipped])
        elif self.loss_type == 'NCE':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            loss = -F.cross_entropy(logits, best_non_target_class)
        else:
            loss = F.cross_entropy(logits, labels)
        return loss

class ModelBase(nn.Module):

    def __init__(self,
                 config, pyg_data, device, logger,
                 with_relu=True, with_bias=True, with_bn=False):

        super(ModelBase, self).__init__()
        self.config = config
        self.pyg_data = pyg_data
        self.device = device
        self.logger = logger

        self.with_relu = with_relu
        self.with_bias = with_bias
        self.with_bn = with_bn

        self.x = pyg_data.x.to(self.device)
        self.edge_index = pyg_data.adj_t.to(self.device)
        self.labels = self.pyg_data.y.to(self.device)

        if 'learning_rate' in self.config:
            self.lr = self.config['learning_rate']
        if 'weight_decay' in self.config:
            self.weight_decay = self.config['weight_decay']
        if 'dropout' in self.config:
            self.dropout = self.config['dropout']


    def initialize(self):
        pass

    @abstractmethod
    def _forward(self, x, mat, weight=None):
        pass

    @abstractmethod
    def forward(self, x, mat, weight=None):
        pass


    def fit(self, pyg_data, adj_t=None, train_iters=None, patience=None, initialize=True, verbose=False):

        if initialize:
            self.initialize()

        if train_iters is None:
            train_iters = self.config['num_epochs']
        if patience is None:
            patience = self.config['patience']

        # self.logger.debug(f"total training epochs: {train_iters}, patience: {patience}")

        self.x = pyg_data.x.to(self.device)
        if adj_t is None:
            self.edge_index = pyg_data.adj_t.to(self.device)
        else:
            self.edge_index = adj_t.to(self.device)


        if patience < train_iters:
            self._train_with_early_stopping(train_iters, patience, verbose)
        else:
            self._train_with_val(train_iters, verbose)

    def _train_with_val(self, train_iters, verbose):
        if verbose:
            self.logger.debug('=== training gcn model ===')

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        best_weight = None

        train_mask = self.pyg_data.train_mask
        val_mask = self.pyg_data.val_mask

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.x, self.edge_index)
            loss_train = F.nll_loss(output[train_mask], self.labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                self.logger.debug(f'Epoch {i:04d}, training loss: {loss_train.item():.5f}')

            self.eval()
            output = self.forward(self.x, self.edge_index)
            loss_val = F.nll_loss(output[val_mask], self.labels[val_mask])
            acc_val = _utils.accuracy(output[val_mask], self.labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                best_weight = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                best_weight = deepcopy(self.state_dict())

        if verbose:
            self.logger.debug('=== picking the best model according to the performance on validation ===')
        if best_weight is not None:
            self.load_state_dict(best_weight)

    def _train_with_early_stopping(self, train_iters, patience, verbose):
        if verbose:
            self.logger.debug(f'=== training {self.__class__.__name__} model ===')

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100
        best_weight = None

        train_mask = self.pyg_data.train_mask
        val_mask = self.pyg_data.val_mask

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.x, self.edge_index)
            loss_train = F.nll_loss(output[train_mask], self.labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                self.logger.debug(f'Epoch {i:04d}, training loss: {loss_train.item():.5f}')

            self.eval()
            output = self.forward(self.x, self.edge_index)

            loss_val = F.nll_loss(output[val_mask], self.labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                best_weight = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             self.logger.debug(f'=== early stopping at {i:04d}, loss_val = {best_loss_val:.5f} ===')
        if best_weight is not None:
            self.load_state_dict(best_weight)

    def test(self, test_mask, verbose=True):
        self.eval()
        output = self.predict()
        loss_test = F.nll_loss(output[test_mask], self.labels[test_mask])
        acc_test = _utils.accuracy(output[test_mask], self.labels[test_mask])
        if verbose:
            self.logger.debug(f"Test set results: loss= {loss_test.item():.4f}, accuracy= {float(acc_test):.4f}")
        return float(acc_test)

    def predict(self, x=None, edge_index=None):

        self.eval()
        if x is None and edge_index is None:
            return self.forward(self.x, self.edge_index).detach()
        else:
            assert (type(edge_index) is torch.Tensor and edge_index.size(0) == 2) or (type(edge_index) is SparseTensor)
            assert type(x) is torch.Tensor
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            return self.forward(x, edge_index).detach()
