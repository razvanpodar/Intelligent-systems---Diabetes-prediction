from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import csv

target = []
data = []
target_names = []
features_names = []

# Read csv using pandas

df = pd.read_csv('diabetes_dataset.csv', converters = {"preg":float, "plas":float, "pres":float, "skin":float, "insu":float, "mass":float, "pedi":float, "age":float})

for col in df:
    features_names.append(col)

del features_names[-1]
data = df.loc[:, features_names]

target_names = df['class'].unique()

for row in df['class']:
    if(row == target_names[0]):
        target.append(1)
    else:
        target.append(0)

# Convert from dataframe to array

data = data.values
data1 = []

for row in data:
    s = row
    for i in row:
        data1.append(i)

data = data1

print(target_names)
print(target)
print(features_names)
print(data)

# Test rows

test1 = 766
test2 = 767

# Training data

train_target = np.delete(target, test2)
train_target = np.delete(train_target, test1)

train_data = np.delete(data, test2)
train_data = np.delete(train_data, test1)

# Testing data

test_target = []
test_target.append(target[test1])
test_target.append(target[test2])

test_data = []
test_data.append(data[test1])
test_data.append(data[test2])

'''
# Decision tree classifier

classif = tree.DecisionTreeClassifier()
classif.fit(train_data, train_target)

print (test_target)
print (classif.predict(test_data))

'''