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

test_idx = [768, 769]


