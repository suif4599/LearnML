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
tree = DecisionTree(namedtuple('Bunch', ['data', 'target'])(train_data, train_target))
cnt = 0
for data, target in zip(test_data, test_target):
    p = tree.predict(data)
    if p == target:
        cnt += 1
print(f"My acc: {cnt / test_data.shape[0]}")


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(train_data, train_target)
print(f"Sklearn acc: {clf.score(test_data, test_target)}")
