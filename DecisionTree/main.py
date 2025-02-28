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
# np.random.shuffle(data)
# np.random.shuffle(target)
# train_data = data[:10, :3]
# train_target = target[:10]
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
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(train_data, train_target)
print(f"Sklearn acc: {clf.score(test_data, test_target):.3f}")
# # show the tree
# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 10))
# plot_tree(clf, filled=True) 
# plt.show()
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_data, train_target)
print(f"RandomForest acc: {clf.score(test_data, test_target):.3f}")

# from autogluon.tabular import TabularDataset, TabularPredictor
# train_data = TabularDataset(train_data, train_target)
# test_data = TabularDataset(test_data, test_target)
# predictor = TabularPredictor(label='target').fit(train_data)
# print(f"AutoGluon acc: {predictor.evaluate(test_data):.3f}")