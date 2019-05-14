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

csv_object = csv.reader(open('diabetes_dataset.csv', 'r'))
header = next(csv_object)

dataset = []
for row in csv_object:
    dataset.append(row)
#data = np.array(data)

'''
iris.target[0] - tested_negative
iris.data[0]   - all the info from first
target names - class
feature names - preg, age, etc.
'''

test_idx = [768, 769]

target = []
data = []
target_names = []
feature_names = []

#print (dataset)
for row in dataset:
    #if row
    #target_names.append()
    #feature_names.append()

    target.append(row[-1])
    row_data = row
    del row_data[-1]
    data.append(row_data)


#target = dataset[9]

#print (target_names)
print (target)
print (data)
'''
print(header)
for row in data:
    print(row)

'''
