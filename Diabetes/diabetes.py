from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.datasets import load_iris
import numpy as np
import csv

iris = load_iris()
print (iris.feature_names)
print (iris.target_names)
print (iris.data[0])
print (iris.target[0])

target = []
data = []
target_names = []
feature_names = []

# Get data from csv
csv_object = csv.reader(open('diabetes_dataset.csv', 'r'))
header = next(csv_object)

dataset = []
for row in csv_object:
    dataset.append(row)
#data = np.array(dataset)

# Separate data
feat_names = header
del feat_names[-1]
feature_names =  feat_names

target_aux = []

for row in dataset:
    target_aux.append(row[-1])
    row_data = row
    del row_data[-1]
    data.append(row_data)

target_names = target_aux
target_names = list(dict.fromkeys(target_names))

for row in target_aux:
    if(row == target_names[0]):
        target.append(1)
    else:
        target.append(0)

print (target_names)
print (target)
print (feature_names)
print (data)

# test_idx = [766, 767]
test_idx = slice(766, 767)

test1 = 766
test2 = 767

# Training data
# train_target = np.delete(target, test_idx)

train_target = np.delete(target, test1)
train_target = np.delete(target, test2)

# train_data = np.delete(data, test_idx, axis = 0)

train_data = np.delete(data, test1, axis = 0)
train_data = np.delete(data, test2, axis = 0)

# Testing data
# test_target = target[test_idx]

test_target = []
test_target.append(target[test1])
test_target.append(target[test2])

# test_data = data[test_idx]

test_data = []
test_data.append(data[test1])
test_data.append(data[test2])

# Decision tree classifier
classif = tree.DecisionTreeClassifier()
classif.fit(train_data, train_target)

print (test_target)
print (classif.predict(test_data))





