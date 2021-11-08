# CS464 CS 464, Introduction to Machine Learning
# Homework 2, Question 2: Linear & Polynomial Regression
# Mustafa Goktan Gudukbay, 21801740, Section 2

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

#Get root, training and test file names
root = ""
house_features_file_name = "question-2-features.csv"

#Open question-2-features.csv and save features to the house_features file
csv_path = os.path.join(root, house_features_file_name)
with open(csv_path, 'r') as csvfile:
    lines = csv.reader(csvfile)
    list_of_rows = list(lines)

    del list_of_rows[0]

    house_features = np.array(list_of_rows).astype(float)

#Open question-2-labels.csv and save features to the house_features file
house_labels_file_name = "question-2-labels.csv"
csv_path = os.path.join(root, house_labels_file_name)
with open(csv_path, 'r') as csvfile:
    lines = csv.reader(csvfile)
    list_of_rows = list(lines)

    del list_of_rows[0]

    house_labels = np.array(list_of_rows).astype(float)


#adding 1s as a column
house_features = np.insert(house_features, 0, np.ones(len(house_features)), axis=1)

#rank
print("Rank of X_transposed * X: ", np.linalg.matrix_rank(np.dot(np.transpose(house_features), house_features)), "\n")

#column of ones and sqftliving feature
feature_columns = house_features[:,0:2]

feature_columns_transpose = np.transpose(feature_columns)

train_weight = np.dot(np.dot(np.linalg.inv(np.dot(feature_columns_transpose, feature_columns)), feature_columns_transpose), house_labels)

print("Part 2.3\nWeights: ", "w0: ", train_weight[0][0], ", w1: ", train_weight[1][0])

x_values = [0, 15000]
y_values = [(train_weight[0][0]), 15000*train_weight[1][0] + train_weight[0][0]]
plt.plot(x_values, y_values, color = "black")

house_features_flattened = house_features[:, 1:2].flatten()
house_labels_flattened =  house_labels.flatten()
plt.scatter(house_features_flattened, house_labels_flattened, color = "red")

plt.legend(('Predicted Price', 'Real Price'), loc="upper right")

plt.title("Relation Between House Prices and Living Room Size")
plt.xlabel("Living Room Size (square feet)")
plt.ylabel("Price")
plt.show()

#Calculating MSE (mean squared error)
predicted_array = house_features_flattened*train_weight[1][0] + train_weight[0][0]

total_loss = (1/len(house_labels_flattened))*np.sum(np.square(predicted_array-house_labels_flattened))
print("MSE: ", total_loss, "\n\n-------------------------------")

#2.4
#adding squared house features as a column
feature_columns = np.insert(feature_columns, 2, np.square(house_features[:, 1:2].flatten()), axis=1)

feature_columns_transpose = np.transpose(feature_columns)

train_weight = np.dot(np.dot(np.linalg.inv(np.dot(feature_columns_transpose, feature_columns)), feature_columns_transpose), house_labels)

print("Part 2.4\nWeights: ", "w0: ", train_weight[0][0], ", w1: ", train_weight[1][0], ", w2: ", train_weight[2][0])

x_values = []
y_values = []
for i in range (0, 15000):
    x_values.append(i)
    y_values.append((i**2)*train_weight[2][0] + i*train_weight[1][0] + train_weight[0][0])

plt.plot(x_values, y_values, color = "black")

plt.scatter(house_features_flattened, house_labels_flattened, color = "red")

plt.legend(('Predicted Price', 'Real Price'), loc="upper right")

plt.title("Relation Between House Prices and Living Room Size")
plt.xlabel("Living Room Size (square feet)")
plt.ylabel("Price")
plt.show()

#Calculating MSE (mean squared error)
predicted_array = (house_features_flattened**2)*train_weight[2][0] + house_features_flattened*train_weight[1][0] + train_weight[0][0]

total_loss = (1/len(house_labels_flattened))*np.sum(np.square(predicted_array-house_labels_flattened))
print("MSE: ", total_loss)
