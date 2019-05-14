from sklearn.neural_network import MLPClassifier
import numpy as np
import csv

'''
with open("diabetes_dataset.csv", mode = "r") as diabetes_dataset:
    diabetes = csv.DictReader(diabetes_dataset)
    line_count = 0
    for row in diabetes:
        #print(row)
        if line
        print(f'Column names are {", ".join(row)}')
'''

csv_object = csv.reader(open('diabetes_dataset.csv', 'r'))
header = next(csv_object)

data = []
for row in csv_object:
    data.append(row)
data = np.array(data)

print(header)
for row in data:
    print(row)