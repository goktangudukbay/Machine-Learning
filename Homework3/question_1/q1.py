import os
import csv
import numpy as np
from sklearn import svm

import matplotlib.pyplot as plt


#Get root, training and test file names
root = ""
breast_cancer_path = "breast_cancer.csv"

#Open x_train.csv and calculate the likelihood array
csv_path = os.path.join(root, breast_cancer_path)


#Open x_train.csv
with open(csv_path, 'r') as csvfile:
    lines = csv.reader(csvfile)
    next(lines)
    breast_cancer_features = [[int(j) for j in i] for i in lines]

breast_cancer_features = np.array(breast_cancer_features)
breast_cancer_features = breast_cancer_features.astype(int)
breast_cancer_classes = breast_cancer_features[:, -1]
breast_cancer_features = np.delete(breast_cancer_features, -1, 1)

#train
breast_cancer_train_features = breast_cancer_features[0:500]
breast_cancer_train_labels = breast_cancer_classes[0:500]

#test
breast_cancer_test_features = breast_cancer_features[500:]
breast_cancer_test_labels = breast_cancer_classes[500:]

#Part 1
print("\nPart 1\n")

mean_accuracy_array = np.zeros(6)
for c in range(6):
    accuracy = 0
    model = svm.SVC(kernel='linear', C=0.001*(10**c))
    for i in range(10):
        vald_start = i*50
        vald_end = vald_start + 50

        k_train_features = np.concatenate((breast_cancer_train_features[0:vald_start],
                                          breast_cancer_train_features[vald_end:]))

        k_train_labels = np.concatenate((breast_cancer_train_labels[0:vald_start], breast_cancer_train_labels[vald_end:]))

        model.fit(k_train_features, k_train_labels)

        if(vald_end == 500):
            predicted_vald_labels = model.predict(breast_cancer_train_features[vald_start:])
        else:
            predicted_vald_labels = model.predict(breast_cancer_train_features[vald_start:vald_end])

        accuracy += np.sum(predicted_vald_labels == breast_cancer_train_labels[vald_start:vald_end])/50

    mean_accuracy_array[c] = accuracy/10

plt.plot([0.001, 0.01, 0.1, 1, 10, 100], mean_accuracy_array, c= "blue")
plt.ylabel('Mean Accuracy')
plt.xlabel('C Value')
plt.title('Mean Accuracy vs C Value')
plt.show()

#find best model
best_model_c_index = 0
for i in range(1, 6):
    if mean_accuracy_array[i] > mean_accuracy_array[best_model_c_index]:
        best_model_c_index = i

print("\nBest C Value: ", 0.001*(10**best_model_c_index))

best_model = svm.SVC(kernel='linear', C=0.001*(10**best_model_c_index))
best_model.fit(breast_cancer_train_features, breast_cancer_train_labels)
predicted_test_labels = best_model.predict(breast_cancer_test_features)

true_positives = np.sum(np.where((predicted_test_labels == 1) & (breast_cancer_test_labels == 1), 1, 0))
false_positives = np.sum(np.where((predicted_test_labels == 1) & (breast_cancer_test_labels == 0), 1, 0))
true_negative = np.sum(np.where((predicted_test_labels == 0) & (breast_cancer_test_labels == 0), 1, 0))
false_negative = np.sum(np.where((predicted_test_labels == 0) & (breast_cancer_test_labels == 1), 1, 0))
accuracy = (true_positives + true_negative)/len(predicted_test_labels)
precision = true_positives/(true_positives+false_positives)
recall = true_positives/(true_positives + false_negative)
negative_predictive_value = true_negative / (true_negative + false_negative)
false_positive_rate = false_positives / (false_positives + true_negative)
false_discovery_rate = false_positives / (false_positives + true_positives)
f1_score = 2 * (precision * recall) / (precision + recall)
f2_score = 5 * (precision * recall) / ((4 * precision) + recall)

print("\nConfusion Matrix\n")

print("\t\t\t    Actual\nClassifier    \t\t\t Malignant Tumor   Benign Tumor\n\t    Malignant Tumor \t", true_positives, "  \t\t\t ", false_positives, "\n\t    Benign Tumor \t\t", false_negative, "  \t\t\t ", true_negative)

print("Accuracy: ", accuracy, "\nPrecision: ", precision, "\nRecall: ", recall)
print("Negative Predictive Values: ", negative_predictive_value, "\nFalse Positive Rate: ", false_positive_rate)
print("False Discovery Rate: ", false_discovery_rate)
print("F1 Score: ", f1_score, "\nF2 Score: ", f2_score, "\n\n")


#-----------------------------------------------------

#Part 2
print("\n\n\n\n----------------------------------------\nPart 2\n")

mean_accuracy_array = np.zeros(5)
gammas = [1/16, 1/8, 1/4, 1, 2]
for gamma_index in range(5):
    accuracy = 0
    model = svm.SVC(kernel='rbf', gamma=gammas[gamma_index])
    for i in range(10):
        vald_start = i*50
        vald_end = vald_start + 50

        k_train_features = np.concatenate((breast_cancer_train_features[0:vald_start],
                                          breast_cancer_train_features[vald_end:]))

        k_train_labels = np.concatenate((breast_cancer_train_labels[0:vald_start], breast_cancer_train_labels[vald_end:]))

        model.fit(k_train_features, k_train_labels)

        if(vald_end == 500):
            predicted_vald_labels = model.predict(breast_cancer_train_features[vald_start:])
        else:
            predicted_vald_labels = model.predict(breast_cancer_train_features[vald_start:vald_end])

        accuracy += np.sum(predicted_vald_labels == breast_cancer_train_labels[vald_start:vald_end])/50

    mean_accuracy_array[gamma_index] = accuracy/10

plt.plot(gammas, mean_accuracy_array, c= "blue")
plt.ylabel('Mean Accuracy')
plt.xlabel('Gamma Value')
plt.title('Mean Accuracy vs Gamma Value')
plt.show()

#find best model
best_model_gamma_index = 0
for i in range(1, 5):
    if mean_accuracy_array[i] > mean_accuracy_array[best_model_gamma_index]:
        best_model_gamma_index = i

print("\nBest Gamma Value: ", gammas[best_model_gamma_index])


model = svm.SVC(kernel='rbf', gamma=gammas[best_model_gamma_index])
best_model.fit(breast_cancer_train_features, breast_cancer_train_labels)
predicted_test_labels = best_model.predict(breast_cancer_test_features)

true_positives = np.sum(np.where((predicted_test_labels == 1) & (breast_cancer_test_labels == 1), 1, 0))
false_positives = np.sum(np.where((predicted_test_labels == 1) & (breast_cancer_test_labels == 0), 1, 0))
true_negative = np.sum(np.where((predicted_test_labels == 0) & (breast_cancer_test_labels == 0), 1, 0))
false_negative = np.sum(np.where((predicted_test_labels == 0) & (breast_cancer_test_labels == 1), 1, 0))
accuracy = (true_positives + true_negative)/len(predicted_test_labels)
precision = true_positives/(true_positives+false_positives)
recall = true_positives/(true_positives + false_negative)
negative_predictive_value = true_negative / (true_negative + false_negative)
false_positive_rate = false_positives / (false_positives + true_negative)
false_discovery_rate = false_positives / (false_positives + true_positives)
f1_score = 2 * (precision * recall) / (precision + recall)
f2_score = 5 * (precision * recall) / ((4 * precision) + recall)

print("\nConfusion Matrix\n")

print("\t\t\t    Actual\nClassifier    \t\t\t Malignant Tumor   Benign Tumor\n\t    Malignant Tumor \t", true_positives, "  \t\t\t ", false_positives, "\n\t    Benign Tumor \t\t", false_negative, "  \t\t\t ", true_negative)

print("Accuracy: ", accuracy, "\nPrecision: ", precision, "\nRecall: ", recall)
print("Negative Predictive Values: ", negative_predictive_value, "\nFalse Positive Rate: ", false_positive_rate)
print("False Discovery Rate: ", false_discovery_rate)
print("F1 Score: ", f1_score, "\nF2 Score: ", f2_score, "\n\n")