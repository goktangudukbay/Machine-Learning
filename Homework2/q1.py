# CS464 CS 464, Introduction to Machine Learning
# Homework 2, Question 1: Principal Component Analysis (PCA) & Digits
# Mustafa Goktan Gudukbay, 21801740, Section 2

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

#Get root, training and test file names
root = ""
digits_features_file_name = "digits.csv"

#Open digits.csv and save features to the digits_features file
csv_path = os.path.join(root, digits_features_file_name)
with open(csv_path, 'r') as csvfile:
    lines = csv.reader(csvfile)
    list_of_rows = list(lines)

    del list_of_rows[0]
    for row in list_of_rows:
        del row[0]

    #10000 x 784
    digits_features = np.array(list_of_rows).astype(float)

#mean centering the data
avg_x = np.sum(digits_features, axis=0)
avg_x = avg_x/len(digits_features)
digits_features = digits_features - avg_x

#covariance matrix
digits_features_transpose = np.transpose(digits_features)
covariance_matrix = np.dot(digits_features_transpose, digits_features)

#eigenvalues and eigenvectors
landa, v = np.linalg.eig(covariance_matrix)
landa = np.real(landa)
v = np.real(v)

eigen_values_descending = landa
eigen_vectors = v

#Question 1.1
#Reporting proportion of variance for first 10 principal components
sum_of_variances = np.sum(eigen_values_descending)
proportion_of_variance_10 = np.zeros(10)

fig=plt.figure(figsize=(28, 28), constrained_layout="True")
fig.tight_layout()
columns = 4
rows = 3

for i in range(10):
    proportion_of_variance_10[i] = eigen_values_descending[i]/sum_of_variances
    fig.add_subplot(rows, columns, i + 1)
    plt.imshow(eigen_vectors[:, [i]].reshape(28, 28), cmap='gray')
    title = "PCA Component " + str(i + 1)
    plt.title(title)

print("Part 1: Proportions of Variance Explained ", "\n")
for i in range(len(proportion_of_variance_10)):
    print("Component: ", i+1, ", PVE: ", proportion_of_variance_10[i])

plt.show()
print("\n\n")
#Question 1.2
#Obtain first k principal components and report PVE for k in {8, 16, 32, 64, 128, 256}
#Plot k vs. PVE and comment on it

print("Part 1.2: \n\n")
sums_for_k_values = []
for i in range(6):
    proportion_of_variance = np.zeros(2**(3+i))
    for j in range(2**(3+i)):
        proportion_of_variance[j] = eigen_values_descending[j] / sum_of_variances
    print("k: ", (2**(3+i), ", Proportion of Variances Explained for Each Component: ", proportion_of_variance))
    x_array = np.full(2**(3+i), 2**(3+i))
    plt.scatter(x_array, proportion_of_variance)

    sums_for_k_values.append(np.sum(proportion_of_variance))
    print("Total Proportion of Variances Explained for All the Components: ", sums_for_k_values[i])

plt.title("Proportion of Variances Explained by Each Component for Different k Values")
plt.xlabel("k")
plt.ylabel("Proportion of Variance Explained")
plt.show()

for i in range(6):
    #plotted total proportion of variances for each k value
    plt.scatter(2**(3+i), sums_for_k_values[i])

plt.plot([8, 16, 32, 64, 128, 256], sums_for_k_values)
plt.title("Total Proportion of Variances Explained for Different k Values")
plt.xlabel("k")
plt.ylabel("Total Proportion of Variance Explained")
plt.show()

#Question 1.3

#first image
first_image_features = digits_features[0]

k_values = [1, 3, 5, 10, 50, 100, 200, 300]

#projected feature array, each row corresponds to the projections if first image with top 1, 3, 5, 10, 50, 100, 200, 300 eigen vectors
projections = []

#analyzing the features, projecting them
for k in k_values:
    eigen_vector_k = eigen_vectors[:, 0:k]
    #multiply first image features with eigen vector 1
    projections.append(np.dot(first_image_features, eigen_vector_k))

#reconstruct the image for each feature
reconstruct = []

for i in range(8):
    eigen_vector_k = eigen_vectors[:, 0:k_values[i]]
    eigen_vector_k_transpose = np.transpose(eigen_vector_k)
    reconstructed_image = np.dot(projections[i], eigen_vector_k_transpose)
    reconstructed_image = np.add(reconstructed_image, avg_x)
    reconstruct.append(reconstructed_image)


fig=plt.figure(figsize=(28, 28), constrained_layout="True")
columns = 3
rows = 3
fig.add_subplot(rows, columns, 1)
plt.imshow((first_image_features+avg_x).reshape(28, 28), cmap='gray')
plt.title("Original Image")
for i in range(1, 9):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(reconstruct[i-1].reshape(28, 28), cmap='gray')
    title = "Reconstructed Image, k: " + str(k_values[i-1])
    plt.title(title)

plt.show()


