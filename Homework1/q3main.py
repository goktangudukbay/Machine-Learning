import os
import csv
import numpy as np
import pathlib

root = pathlib.Path().absolute()

x_train = "x_train.csv"
y_train = "y_train.csv"
x_test = input("Enter test feature file name: ")
y_test = input("Enter test label file name: ")

#Open x_train.csv and calculate the likelihood array
csv_path = os.path.join(root, x_train)

#Open x_train.csv
with open(csv_path, 'r') as csvfile:
    lines = csv.reader(csvfile)
    x_train_features = [[int(j) for j in i] for i in lines]

#Open y_train.csv
csv_path = os.path.join(root, y_train)
with open(csv_path, 'r') as csvfile:
    lines = csv.reader(csvfile)
    y_train_labels = [int(i[0]) for i in lines]

#calcuating prior for spam mail
prior_spam = np.sum(y_train_labels) / len(y_train_labels)

y_train_labels = np.array(y_train_labels)
x1 = np.array(x_train_features)

#calculating the likelihood array for spam
labelBooleanSpam = np.where(y_train_labels > 0, True, False)
word_occurences_spam = x1[labelBooleanSpam].sum(axis=0)
likelihood_array_spam =  np.divide(word_occurences_spam, sum(word_occurences_spam)) 

#calculating the likelihood array for normal
labelBooleanNormal = np.where(y_train_labels == 0, True, False)
word_occurences_normal = x1[labelBooleanNormal].sum(axis=0)
likelihood_array_normal = np.divide(word_occurences_normal, sum(word_occurences_normal))

#Test the data
#Open x_test.csv and create the labels
csv_path = os.path.join(root, x_test)

#Open x_test.csv
with open(csv_path, 'r') as csvfile:
    lines = csv.reader(csvfile)
    x_test_features = [[int(j) for j in i] for i in lines]

x_test_features = np.array(x_test_features)

normal_probability = np.where((x_test_features < np.finfo(np.float64).eps) & (likelihood_array_normal < np.finfo(np.float64).eps), 0, 
                           x_test_features * np.log(likelihood_array_normal))

spam_probability = np.where((x_test_features < np.finfo(np.float64).eps) & (likelihood_array_spam < np.finfo(np.float64).eps), 0, 
                          x_test_features * np.log(likelihood_array_spam))

temp1 = np.sum(normal_probability, axis = 1) + np.log(1 -prior_spam)
temp2 = np.sum(spam_probability, axis = 1) + np.log(prior_spam)

test_label = np.where(temp1 >= temp2, 0, 1)

#Calculate accuracy
#Open y_test.csv
csv_path = os.path.join(root, y_test)

#Open x_test.csv
with open(csv_path, 'r') as csvfile:
    lines = csv.reader(csvfile)
    y_test_labels = [int(i[0]) for i in lines]

y_test_labels = np.array(y_test_labels)

print("Q3.2\n")

#Accuracy
print("Accuracy: ", 100*np.count_nonzero(y_test_labels == test_label)/np.size(y_test_labels), "%")

#Confusion Matrix
print("Confusion Matrix\n")
truePositive = np.count_nonzero((test_label == 1) & (y_test_labels == 1))
trueNegative = np.count_nonzero((test_label == 0) & (y_test_labels == 0))
falsePositive = np.count_nonzero((test_label == 1) & (y_test_labels == 0))
falseNegative = np.count_nonzero((test_label == 0) & (y_test_labels == 1))
print("\t\t\t  Actual\nClassifier    \tSpam    Normal\n\t    Spam \t", truePositive, "   ", falsePositive, "\n\t    Normal \t", falseNegative, "   ", trueNegative)

#Total Number of wrong predictions
print("\nWrong Predictions: ", falsePositive + falseNegative)

print("\n-------------------------------------------\n")

#Part 3.3, Laplace Smoothing
#calculating the likelihood array for spam
word_occurences_spam = x1[labelBooleanSpam].sum(axis=0, initial = 1)
likelihood_array_spam =  np.divide(word_occurences_spam, sum(word_occurences_spam)) 

#calculating the likelihood array for normal
word_occurences_normal = x1[labelBooleanNormal].sum(axis=0, initial = 1)
likelihood_array_normal = np.divide(word_occurences_normal, sum(word_occurences_normal)) 

normal_probability = x_test_features * np.log(likelihood_array_normal)
spam_probability = x_test_features * np.log(likelihood_array_spam)

temp1 = np.nansum(normal_probability, axis = 1) + np.log(1 -prior_spam)
temp2 = np.nansum(spam_probability, axis = 1) + np.log(prior_spam)

test_label = np.where(temp1 >= temp2, 0, 1)

print("Q3.3")

#Accuracy
print("Accuracy: ", 100*np.count_nonzero(y_test_labels == test_label)/np.size(y_test_labels), "%")

#Confusion Matrix
print("Confusion Matrix\n")
truePositive = np.count_nonzero((test_label == 1) & (y_test_labels == 1))
trueNegative = np.count_nonzero((test_label == 0) & (y_test_labels == 0))
falsePositive = np.count_nonzero((test_label == 1) & (y_test_labels == 0))
falseNegative = np.count_nonzero((test_label == 0) & (y_test_labels == 1))
print("\t\t\t  Actual\nClassifier    \tSpam    Normal\n\t    Spam \t", truePositive, "   ", falsePositive, "\n\t    Normal \t", falseNegative, "   ", trueNegative)

#Total Number of wrong predictions
print("\nWrong Predictions: ", falsePositive + falseNegative)

print("\n-------------------------------------------\n")

# Bernouilli Naive Bayes Model
#calculating the likelihood array for spam
word_occurences_spam = np.count_nonzero(x1[labelBooleanSpam], axis=0)
likelihood_array_spam =  np.divide(word_occurences_spam, sum(labelBooleanSpam)) 

#calculating the likelihood array for normal
word_occurences_normal = np.count_nonzero(x1[labelBooleanNormal], axis=0)
likelihood_array_normal = np.divide(word_occurences_normal, sum(labelBooleanNormal)) 

normal_probability = np.where(x_test_features != 0, likelihood_array_normal, (1-likelihood_array_normal))
spam_probability = np.where(x_test_features != 0, likelihood_array_spam, (1-likelihood_array_spam))

temp1 = np.log(np.prod(normal_probability, axis = 1)) + np.log(1-prior_spam)
temp2 = np.log(np.prod(spam_probability, axis = 1)) + np.log(prior_spam)

test_label = np.where(temp1 >= temp2, 0, 1)

print("Q3.4")

#Accuracy
print("Accuracy: ", 100*np.count_nonzero(y_test_labels == test_label)/np.size(y_test_labels), "%")

#Confusion Matrix
print("Confusion Matrix\n")
truePositive = np.count_nonzero((test_label == 1) & (y_test_labels == 1))
trueNegative = np.count_nonzero((test_label == 0) & (y_test_labels == 0))
falsePositive = np.count_nonzero((test_label == 1) & (y_test_labels == 0))
falseNegative = np.count_nonzero((test_label == 0) & (y_test_labels == 1))
print("\t\t\t  Actual\nClassifier    \tSpam    Normal\n\t    Spam \t", truePositive, "   ", falsePositive, "\n\t    Normal \t", falseNegative, "   ", trueNegative)

#Total Number of wrong predictions
print("\nWrong Predictions: ", falsePositive + falseNegative)

print("\n-------------------------------------------\n")