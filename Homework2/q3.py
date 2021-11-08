# CS464 CS 464, Introduction to Machine Learning
# Homework 2, Question 3: Logistic Regression
# Mustafa Goktan Gudukbay, 21801740, Section 2

import os
import csv
import numpy as np
import math


#Get root, training and test file names
root = ""
train_features_file_name = "question-3-features-train.csv"

#Open question-3-features-train.csv and save features to the train_features variable
csv_path = os.path.join(root, train_features_file_name)
with open(csv_path, 'r') as csvfile:
    lines = csv.reader(csvfile)
    list_of_rows = list(lines)

    del list_of_rows[0]

    train_features = np.array(list_of_rows).astype(float)

#Open question-3-labels-train.csv and save labels to the train_labels variable
train_labels_file_name = "question-3-labels-train.csv"
csv_path = os.path.join(root, train_labels_file_name)
with open(csv_path, 'r') as csvfile:
    lines = csv.reader(csvfile)
    list_of_rows = list(lines)

    del list_of_rows[0]

    train_labels = np.array(list_of_rows).astype(int)
    train_labels = train_labels.flatten()

#normalizing last column
min_amount = np.min(train_features[:, [-1]])
max_amount = np.max(train_features[:, [-1]])
max_min_difference = max_amount - min_amount

for i in range(len(train_features)):
    train_features[i][-1] = (train_features[i][-1] - min_amount)/max_min_difference


#Initalizing weights to zero
weights = np.zeros(len(train_features[0]) + 1)

#step size
step_size = 0.0001
train_features_ones_column_added = np.insert(train_features, 0, np.ones(len(train_features)), axis=1)
for i in range(1000):
    sums = np.dot(train_features_ones_column_added, weights)
    probability_values = np.exp(sums) / (np.exp(sums) + 1)
    value = train_labels - probability_values
    feature_multiplication = np.dot(value, train_features_ones_column_added)
    sums_columns = feature_multiplication
    sums_columns = sums_columns * step_size
    weights = sums_columns + weights

#testing the results
#Open question-3-features-test.csv and save features to the test_features variable
test_features_file_name = "question-3-features-test.csv"
csv_path = os.path.join(root, test_features_file_name)
with open(csv_path, 'r') as csvfile:
    lines = csv.reader(csvfile)
    list_of_rows = list(lines)

    del list_of_rows[0]

    test_features = np.array(list_of_rows).astype(float)
    # normalizing last column
    min_amount = np.min(test_features[:, [-1]])
    max_amount = np.max(test_features[:, [-1]])
    max_min_difference = max_amount - min_amount

    for i in range(len(test_features)):
        test_features[i][-1] = (test_features[i][-1] - min_amount) / max_min_difference

#Open question-3-labels-test.csv and save labels to the test_labels variable
test_labels_file_name = "question-3-labels-test.csv"
csv_path = os.path.join(root, test_labels_file_name)
with open(csv_path, 'r') as csvfile:
    lines = csv.reader(csvfile)
    list_of_rows = list(lines)

    del list_of_rows[0]

    test_labels = np.array(list_of_rows).astype(int)
    test_labels = test_labels.flatten()

predictions = np.dot(test_features, weights[1:])
predictions = predictions + weights[0]

predicted_labels = np.where(predictions >= 0, 1, 0)

true_positives = np.sum(np.where((predicted_labels == 1) & (test_labels == 1), 1, 0))
false_positives = np.sum(np.where((predicted_labels == 1) & (test_labels == 0), 1, 0))
true_negative = np.sum(np.where((predicted_labels == 0) & (test_labels == 0), 1, 0))
false_negative = np.sum(np.where((predicted_labels == 0) & (test_labels == 1), 1, 0))
accuracy = (true_positives + true_negative)/len(predicted_labels)
precision = true_positives/(true_positives+false_positives)
recall = true_positives/(true_positives + false_negative)
negative_predictive_value = true_negative / (true_negative + false_negative)
false_positive_rate = false_positives / (false_positives + true_negative)
false_discovery_rate = false_positives / (false_positives + true_positives)
f1_score = 2 * (precision * recall) / (precision + recall)
f2_score = 5 * (precision * recall) / ((4 * precision) + recall)

print("Full Batch Gradient Ascent, ", step_size, "Step Size\n")
print("Accuracy: ", accuracy, "\nPrecision: ", precision, "\nRecall: ", recall)
print("Negative Predictive Values: ", negative_predictive_value, "\nFalse Positive Rate: ", false_positive_rate)
print("False Discovery Rate: ", false_discovery_rate)
print("F1 Score: ", f1_score, "\nF2 Score: ", f2_score, "\n\n")


print("\t\t\t    Actual\nClassifier    \t Fraud   Normal\n\t    Fraud \t", true_positives, "   ", false_positives, "\n\t    Normal \t", false_negative, "   ", true_negative)


#Part 3.2

#mini batch 100

#Initalizing weights with gaussian distribution
weights = np.random.normal(0, 0.01, len(train_features[0]) + 1)

#Index array that will be used for shuffling
indexes = np.arange(len(train_features))

#train_features and train_labels without shuffled
train_features_without_shuffled = train_features_ones_column_added
train_labels_without_shuffled = train_labels

for i in range(1000):
    shuffled_indexes = indexes
    np.random.shuffle(shuffled_indexes)
    train_features = train_features_without_shuffled[shuffled_indexes]
    train_labels = train_labels_without_shuffled[shuffled_indexes]
    division_size = int(len(train_labels)/100 - 1)
    for x in range(division_size):
        ind = shuffled_indexes[(x*100):((x+1)*100)]
        sums = np.dot(train_features[ind], weights)
        probability_values = np.exp(sums) / (np.exp(sums) + 1)
        value = train_labels[ind] - probability_values
        feature_multiplication = np.dot(value, train_features[ind])
        sums_columns = feature_multiplication
        sums_columns = sums_columns * step_size
        weights = sums_columns + weights



predictions = np.dot(test_features, weights[1:])
predictions = predictions + weights[0]

predicted_labels = np.where(predictions >= 0, 1, 0)

true_positives = np.sum(np.where((predicted_labels == 1) & (test_labels == 1), 1, 0))
false_positives = np.sum(np.where((predicted_labels == 1) & (test_labels == 0), 1, 0))
true_negative = np.sum(np.where((predicted_labels == 0) & (test_labels == 0), 1, 0))
false_negative = np.sum(np.where((predicted_labels == 0) & (test_labels == 1), 1, 0))
accuracy = (true_positives + true_negative)/len(predicted_labels)
precision = true_positives/(true_positives+false_positives)
recall = true_positives/(true_positives + false_negative)
negative_predictive_value = true_negative / (true_negative + false_negative)
false_positive_rate = false_positives / (false_positives + true_negative)
false_discovery_rate = false_positives / (false_positives + true_positives)
f1_score = 2 * (precision * recall) / (precision + recall)
f2_score = 5 * (precision * recall) / ((4 * precision) + recall)

print("\n\nMini Batch Gradient Ascent, ", step_size, "Step Size\n")
print("Accuracy: ", accuracy, "\nPrecision: ", precision, "\nRecall: ", recall)
print("Negative Predictive Values: ", negative_predictive_value, "\nFalse Positive Rate: ", false_positive_rate)
print("False Discovery Rate: ", false_discovery_rate)
print("F1 Score: ", f1_score, "\nF2 Score: ", f2_score, "\n\n")


print("\t\t\t    Actual\nClassifier    \t Fraud   Normal\n\t    Fraud \t", true_positives, "   ", false_positives, "\n\t    Normal \t", false_negative, "   ", true_negative)


#Stochastic gradient descent

#Initalizing weights with gaussian distribution
weights = np.random.normal(0, 0.01, len(train_features[0]))

for i in range(1000):
    shuffled_indexes = indexes
    np.random.shuffle(shuffled_indexes)
    train_features = train_features_without_shuffled[shuffled_indexes]
    train_labels = train_labels_without_shuffled[shuffled_indexes]

    for j in range(len(train_labels)):
        weights_features_sum = np.dot(train_features[j], weights)
        probability = (np.exp(weights_features_sum)) / (np.exp(weights_features_sum) + 1)
        #value is y_j - P(Y=1| X, w)
        value = train_labels[j] - probability
        weights = weights + (step_size * (train_features[j]*value))


predictions = np.dot(test_features, weights[1:])
predictions = predictions + weights[0]

predicted_labels = np.where(predictions >= 0, 1, 0)

true_positives = np.sum(np.where((predicted_labels == 1) & (test_labels == 1), 1, 0))
false_positives = np.sum(np.where((predicted_labels == 1) & (test_labels == 0), 1, 0))
true_negative = np.sum(np.where((predicted_labels == 0) & (test_labels == 0), 1, 0))
false_negative = np.sum(np.where((predicted_labels == 0) & (test_labels == 1), 1, 0))
accuracy = (true_positives + true_negative)/len(predicted_labels)
precision = true_positives/(true_positives+false_positives)
recall = true_positives/(true_positives + false_negative)
negative_predictive_value = true_negative / (true_negative + false_negative)
false_positive_rate = false_positives / (false_positives + true_negative)
false_discovery_rate = false_positives / (false_positives + true_positives)
f1_score = 2 * (precision * recall) / (precision + recall)
f2_score = 5 * (precision * recall) / ((4 * precision) + recall)

print("\n\nStochastic Gradient Ascent, ", step_size, "Step Size\n")
print("Accuracy: ", accuracy, "\nPrecision: ", precision, "\nRecall: ", recall)
print("Negative Predictive Values: ", negative_predictive_value, "\nFalse Positive Rate: ", false_positive_rate)
print("False Discovery Rate: ", false_discovery_rate)
print("F1 Score: ", f1_score, "\nF2 Score: ", f2_score, "\n\n")


print("\t\t\t    Actual\nClassifier    \t Fraud   Normal\n\t    Fraud \t", true_positives, "   ", false_positives, "\n\t    Normal \t", false_negative, "   ", true_negative)
