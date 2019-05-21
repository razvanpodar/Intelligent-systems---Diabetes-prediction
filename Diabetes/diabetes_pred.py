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

print(target_names)
print(target)
print(features_names)
print(data1)

# Test rows

test1 = 766
test2 = 767

nrOfTests = 2

# Training data

train_target = target[:-(nrOfTests * len(features_names))]

train_data = data[:-(nrOfTests * len(features_names))]

# Testing data

test_target = []
test_target = target[-(nrOfTests * len(features_names)):]

test_data = []
test_data = data[-(nrOfTests * len(features_names)):]

# Decision tree classifier

classif = tree.DecisionTreeClassifier()
classif.fit(train_data, train_target)

print (test_target)
print (classif.predict(test_data))

# Test classifier's accuracy

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = .5)

classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

print("Acuratete:")
print (accuracy_score(y_test, predictions))

# Neural network

clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (15, ), random_state = 1)
clf.fit(X_train, y_train)

print (clf.predict(test_data))

predictions = clf.predict(X_test)

print (accuracy_score(y_test, predictions))