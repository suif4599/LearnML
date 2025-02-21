import numpy as np
from numpy.typing import NDArray
from sklearn.utils import Bunch
from collections.abc import Callable, Sequence
from typing import TypeAlias

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

    def __init__(self, databaset: Bunch):
        self.data = databaset.data
        self.target = databaset.target
        self.attrs = DecisionTree.gen_attr(self.data, 4)
        self.tree = self.generate(self.data, self.target, self.attrs)

    def generate(self, data: NDArray, target: NDArray, attributes: set[Attribute]) -> Node:
        attributes = attributes.copy()
        if np.all(target == target[0]):
            return Node(lambda x: np.full(x.shape[0], target[0]))
        if not attributes:
            return Node(lambda x: np.full(x.shape[0], np.argmax(np.bincount(target))))
        best_attr = min(attributes, key=lambda attr: DecisionTree.gain(data, target, attr))
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

    @staticmethod
    def gain(data: NDArray, target: NDArray, attr: Attribute) -> float:
        characteristic = attr(data)
        return DecisionTree.entropy(target) - \
            np.sum([DecisionTree.entropy(target[characteristic == i]) * np.sum(characteristic == i) \
                    for i in np.unique(characteristic)]) / data.shape[0]

    @staticmethod
    def entropy(target: NDArray) -> float:
        p = np.bincount(target) / target.shape[0]
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    
    @staticmethod
    def gen_attr(data: NDArray, splits: int) -> set[Attribute]:
        ret = set()
        for i in range(data.shape[1]):
            attr = NamedAttribute(f"Attribute {i}, {np.linspace(data[:, i].min(), data[:, i].max(), splits)}", 
                                  lambda x: np.digitize(x[:, i], np.linspace(data[:, i].min(), data[:, i].max(), splits)) - 1)
            ret.add(attr)
        return ret

