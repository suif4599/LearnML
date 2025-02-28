import numpy as np
from numpy.typing import NDArray
from sklearn.utils import Bunch
from collections.abc import Callable, Sequence
from typing import TypeAlias
from itertools import combinations

Attribute: TypeAlias = Callable[[NDArray], NDArray] # in_shape: (n_samples, n_features) -> out_shape: (n_samples,) int

class NamedAttribute:
    def __init__(self, info: str, attr: Attribute):
        self.info = info
        self.attr = attr

    def __call__(self, data: NDArray) -> NDArray:
        return self.attr(data)
    
    def __str__(self):
        return self.info

class Node:
    def __init__(self, attr: Attribute, child: Sequence['Node'] = ()):
        self.attr = attr
        self.child = child
    
    def pridict(self, data: NDArray) -> int:
        characteristic = int(self.attr(data.reshape((1, data.shape[0]))).reshape((1, )))
        if not self.child:
            return characteristic
        try:
            return self.child[characteristic].pridict(data)
        except IndexError:
            return self.child[-1].pridict(data)

class DecisionTree:
    data: NDArray # shape: (n_samples, n_features)
    target: NDArray # shape: (n_samples,), non-negative integers
    gain: Callable[[NDArray, NDArray, Attribute], float]
    split: Callable[[NDArray, NDArray], set[Attribute]]

    def __init__(self, databaset: Bunch, 
                 gain: Callable[[NDArray, NDArray, Attribute], float] = None,
                 split: Callable[[NDArray, NDArray], set[Attribute]] = None):
        if gain is not None:
            self.gain = gain
        else:
            self.gain = DecisionTree.info_gain
        if split is not None:
            self.split = split
        else:
            self.split = DecisionTree.mean_split(4)
        self.data = databaset.data
        self.target = databaset.target
        self.attrs = self.split(self.data, self.target)
        self.tree = self.generate(self.data, self.target, self.attrs)

    def generate(self, data: NDArray, target: NDArray, attributes: set[Attribute]) -> Node:
        attributes = attributes.copy()
        if np.all(target == target[0]):
            return Node(lambda x: np.full(x.shape[0], target[0]))
        if not attributes:
            return Node(lambda x: np.full(x.shape[0], np.argmax(np.bincount(target))))
        best_attr = max(attributes, key=lambda attr: self.gain(data, target, attr))
        attributes.remove(best_attr)
        characteristic = best_attr(data)
        return Node(best_attr,
                    tuple(self.generate(data[characteristic == i], target[characteristic == i], attributes) \
                          for i in np.unique(characteristic)))

    def __str__(self):
        def str_node(node: Node, depth: int):
            return '\n'.join([f"{'  ' * depth}{node.attr}"] + \
                             [str_node(child, depth + 1) for child in node.child])
        return str_node(self.tree, 0)
    
    def predict(self, data: NDArray) -> int:
        return self.tree.pridict(data)
    
    def score(self, data: NDArray, target: NDArray) -> float:
        cnt = 0
        for d, tar in zip(data, target):
            p = self.predict(d)
            if p == tar:
                cnt += 1
        return cnt / data.shape[0]

    @staticmethod
    def info_gain(data: NDArray, target: NDArray, attr: Attribute) -> float:
        characteristic = attr(data)
        if np.all(characteristic == characteristic[0]):
            return 0.
        return DecisionTree.entropy(target) - \
            np.sum([DecisionTree.entropy(target[characteristic == i]) * np.sum(characteristic == i) \
                    for i in np.unique(characteristic)]) / data.shape[0]

    @staticmethod
    def entropy(target: NDArray) -> float:
        p = np.bincount(target) / target.shape[0]
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    
    @staticmethod
    def IV(data: NDArray, target: NDArray, attr: Attribute) -> float:
        characteristic = attr(data)
        p = np.bincount(characteristic) / data.shape[0]
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    @staticmethod
    def gain_rate(data: NDArray, target: NDArray, attr: Attribute) -> float:
        characteristic = attr(data)
        if np.all(characteristic == characteristic[0]):
            return 0.
        return DecisionTree.info_gain(data, target, attr) / DecisionTree.IV(data, target, attr)
    
    @staticmethod
    def neg_gini_index(data: NDArray, target: NDArray, attr: Attribute) -> float:
        characteristic = attr(data)
        if np.all(characteristic == characteristic[0]):
            return 0.
        def p(target: NDArray) -> float:
            return np.bincount(target) / target.shape[0]
        return np.sum(np.sum(characteristic == i) * np.sum(p(target[characteristic == i]) ** 2) \
                      for i in np.unique(characteristic)) / data.shape[0]
    
    @staticmethod
    def mean_split(splits: int) -> Callable[[NDArray, NDArray], set[Attribute]]:
        def mean_split_inner(data: NDArray, target: NDArray) -> set[Attribute]:
            ret = set()
            for i in range(data.shape[1]):
                def lambda_(x: NDArray, i=i) -> NDArray:
                    return np.digitize(x[:, i], np.linspace(data[:, i].min(), data[:, i].max(), splits)) - 1
                attr = NamedAttribute(f"Mean split on Attribute {i}, {np.linspace(data[:, i].min(), data[:, i].max(), splits)}", lambda_)
                ret.add(attr)
            return ret
        return mean_split_inner
    
    @staticmethod
    def percentile_split(splits: int) -> Callable[[NDArray, NDArray], set[Attribute]]:
        def percentile_split_inner(data: NDArray, target: NDArray) -> set[Attribute]:
            ret = set()
            for i in range(data.shape[1]):
                def lambda_(x: NDArray, i=i) -> NDArray:
                    return np.digitize(x[:, i], np.percentile(data[:, i], range(0, 101, 100 // splits))) - 1
                attr = NamedAttribute(f"Cluster split on Attribute {i}", lambda_)
                ret.add(attr)
            return ret
        return percentile_split_inner
    
    @staticmethod
    def split_based_on_gain(splits: int, gain: Callable[[NDArray, NDArray, Attribute], float]) -> Callable[[NDArray, NDArray], set[Attribute]]:
        k = 20
        def split_based_on_gain_inner(data: NDArray, target: NDArray) -> set[Attribute]:
            ret = set()
            for i in range(data.shape[1]):
                midpts = (data[:, i][1:] + data[:, i][:-1]) / 2
                percentiles = np.percentile(data[:, i], np.linspace(0, 100, k + 1)[1:-1])
                midpts = np.unique(percentiles)
                cms = tuple(combinations(midpts, splits))
                mapping = []
                for midpt in cms:
                    mapping.append(gain(data, target, 
                                        lambda x: np.digitize(x[:, i], midpt)))
                midpt = cms[np.argmax(mapping)]
                def lambda_(x: NDArray, i=i, midpt=midpt) -> NDArray:
                    return np.digitize(x[:, i], midpt)
                attr = NamedAttribute(f"Split on Attribute {i} > {midpt}", lambda_)
                ret.add(attr)
            return ret
        return split_based_on_gain_inner