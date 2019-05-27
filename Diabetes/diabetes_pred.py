from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import graphviz

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
data1 = data
data2 = []

for row in data:
    s = row
    data2.append(row)

data = data2

print()
print("Possible results(The 2 output nodes of the neural network)")
print(target_names)
print()
print("List of all the results encoded(tested_positive = 0, tested_negative = 1")
print(target)
print()
print("The features(the input nodes in the neural network, 8 features meaning 8 input nodes for the neural network)")
print(features_names)
print()
print("This is the data from which the classifier learns(the values of each feature)")
print(data1)
print()

# Test rows

test1 = 766
test2 = 767

nrOfTests = 2
print("Number of rows for tests")
print(nrOfTests)
print()

# Training data

train_target = target[:-(nrOfTests * len(features_names))]

train_data = data[:-(nrOfTests * len(features_names))]

# Testing data

test_target = []
test_target = target[-(nrOfTests * len(features_names)):]

test_data = []
test_data = data[-(nrOfTests * len(features_names)):]

print("The expected results after tests(last 16 rows of the data set")
print (test_target)
print()

# Decision tree classifier

classif = tree.DecisionTreeClassifier()
classif.fit(train_data, train_target)

print("Decision tree results")
pred = classif.predict(test_data)
print (classif.predict(test_data))

print("Decision tree accuracy")
print (accuracy_score(test_target, pred))
print()

# Neural network

clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (15, ), random_state = 1)
clf.fit(train_data, train_target)

print("Neural network results")
print (clf.predict(test_data))

predictions = clf.predict(test_data)

print("Neural network accuracy")
print (accuracy_score(test_target, predictions))