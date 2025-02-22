from sklearn.datasets import load_iris, load_breast_cancer
from core.decision_tree import DecisionTree
from collections import namedtuple
import numpy as np

# dataset = load_iris()
dataset = load_breast_cancer()
data = dataset.data
target = dataset.target
mask = np.random.rand(data.shape[0]) < 0.8
train_data = data[mask]
train_target = target[mask]
test_data = data[~mask]
test_target = target[~mask]
splits = 3
tree = DecisionTree(namedtuple('Bunch', ['data', 'target'])(train_data, train_target))
print(f"info_gain mean_split acc: {tree.score(test_data, test_target):.3f}")
tree = DecisionTree(namedtuple('Bunch', ['data', 'target'])(train_data, train_target), DecisionTree.gain_rate)
print(f"gain_rate mean_split acc: {tree.score(test_data, test_target):.3f}")
tree = DecisionTree(namedtuple('Bunch', ['data', 'target'])(train_data, train_target), DecisionTree.neg_gini_index)
print(f"gini_index mean_split acc: {tree.score(test_data, test_target):.3f}")
tree = DecisionTree(namedtuple('Bunch', ['data', 'target'])(train_data, train_target), DecisionTree.info_gain, DecisionTree.percentile_split(splits))
print(f"info_gain percentile_split acc: {tree.score(test_data, test_target):.3f}")
tree = DecisionTree(namedtuple('Bunch', ['data', 'target'])(train_data, train_target), DecisionTree.gain_rate, DecisionTree.percentile_split(splits))
print(f"gain_rate percentile_split acc: {tree.score(test_data, test_target):.3f}")
tree = DecisionTree(namedtuple('Bunch', ['data', 'target'])(train_data, train_target), DecisionTree.neg_gini_index, DecisionTree.percentile_split(splits))
print(f"gini_index percentile_split acc: {tree.score(test_data, test_target):.3f}")
tree = DecisionTree(namedtuple('Bunch', ['data', 'target'])(train_data, train_target), DecisionTree.info_gain, DecisionTree.split_based_on_gain(splits, DecisionTree.neg_gini_index))
print(f"info_gain split_based_on_gain acc: {tree.score(test_data, test_target):.3f}")
tree = DecisionTree(namedtuple('Bunch', ['data', 'target'])(train_data, train_target), DecisionTree.gain_rate, DecisionTree.split_based_on_gain(splits, DecisionTree.neg_gini_index))
print(f"gain_rate split_based_on_gain acc: {tree.score(test_data, test_target):.3f}")
tree = DecisionTree(namedtuple('Bunch', ['data', 'target'])(train_data, train_target), DecisionTree.neg_gini_index, DecisionTree.split_based_on_gain(splits, DecisionTree.neg_gini_index))
print(f"gini_index split_based_on_gain acc: {tree.score(test_data, test_target):.3f}")
# print(tree.score(train_data, train_target))

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(train_data, train_target)
print(f"Sklearn acc: {clf.score(test_data, test_target):.3f}")