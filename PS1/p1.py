import numpy as np
import matplotlib.pyplot as plt

# set up the parameters of the class-conditioned Gaussian pdfs
mu_0 = np.array([-1, 1, -1, 1])
mu_1 = np.array([1, 1, 1, 1])

sigma_0 = np.array([[2, -0.5, 0.3, 0],
                    [-0.5, 1, -0.5, 0],
                    [0.3, -0.5, 1, 0],
                    [0, 0, 0, 2]])
sigma_1 = np.array([[1, 0.3, -0.2, 0],
                    [0.3, 2, 0.3, 0],
                    [-0.2, 0.3, 1, 0],
                    [0, 0, 0, 3]])

# set up the parameters of the class priors
p_0 = 0.7
p_1 = 0.3

# generate 10000 samples according to the data distribution
samples = 10000
x = np.random.multivariate_normal(mu_0, sigma_0, samples)
y = np.random.multivariate_normal(mu_1, sigma_1, samples)

data_set = []

# use the class priors to generate the labels
for i in range(samples):
    if np.random.rand() < p_0:
        data_set.append([x[i], 0])
    else:
        data_set.append([y[i], 1])


# part a
# 1.
# specify the minimum expected risk classification rule in the form of a likelihood ratio test
# where the threshold is a function of class priors and fixed loss values for each of the four possible outcomes

threshold = p_1 / p_0
# threshold = (np.log(p_1) - np.log(p_0) + 0.5 * np.log(np.linalg.det(sigma_0) / np.linalg.det(sigma_1))) / \
#             (mu_0.T @ np.linalg.inv(sigma_0) - mu_1.T @ np.linalg.inv(sigma_1)) @ \
#             (mu_0.T @ np.linalg.inv(sigma_0) - mu_1.T @ np.linalg.inv(sigma_1)).T

# Define the likelihood ratio test
def likelihood_ratio_test(x):
    # return np.log(p_1 / p_0) + 0.5 * (x.T @ np.linalg.inv(sigma_0) @ x - x.T @ np.linalg.inv(sigma_1) @ x) + \
    #        0.5 * (mu_0.T @ np.linalg.inv(sigma_0) @ mu_0 - mu_1.T @ np.linalg.inv(sigma_1) @ mu_1)
    ratio_x_0 = np.exp(-0.5 * (x - mu_0).T @ np.linalg.inv(sigma_0) @ (x - mu_0))
    ratio_x_1 = np.exp(-0.5 * (x - mu_1).T @ np.linalg.inv(sigma_1) @ (x - mu_1))
    return np.log(p_1 / p_0) + np.log(ratio_x_0 / ratio_x_1)

# Define the minimum expected risk classification rule
def minimum_expected_risk_classification_rule(x, threshold):
    if likelihood_ratio_test(x) >= threshold:
        return 0
    else:
        return 1
    
# 2.
# implement the classifier and apply it to the data set
# vary the threshold gradually from 0 to infinity and for each value of the threshold, compute the true positive rate and the false positive probabilities
# using these paired values, plot the ROC curve

# Define the ROC curve
def ROC_curve(data_set, threshold):
    true_positive = 0
    false_positive = 0
    for i in range(len(data_set)):
        if data_set[i][1] == 1:
            if minimum_expected_risk_classification_rule(data_set[i][0], threshold) == 1:
                true_positive += 1
        else:
            if minimum_expected_risk_classification_rule(data_set[i][0], threshold) == 1:
                false_positive += 1
    return true_positive / (true_positive + false_positive), false_positive / (true_positive + false_positive)

# Plot the ROC curve
true_positive_rate = []
false_positive_rate = []
for i in range(100):
    threshold = i / 100
    true_positive_rate.append(ROC_curve(data_set, threshold)[0])
    false_positive_rate.append(ROC_curve(data_set, threshold)[1])
plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# 3.
# determine the threshold that minimizes the probability of error
# on the ROC curve, superimpose the true positive and false positive probabilities for this minimum-P(error) threshold
# calculate and report an estimate of the minimum probability of error achievable for this data distribution

# Define the probability of error
def probability_of_error(data_set, threshold):
    error = 0
    for i in range(len(data_set)):
        if data_set[i][1] == 1:
            if minimum_expected_risk_classification_rule(data_set[i][0], threshold) == 0:
                error += 1
        else:
            if minimum_expected_risk_classification_rule(data_set[i][0], threshold) == 1:
                error += 1
    return error / len(data_set)

# Plot the minimum-P(error) threshold
threshold = 0
for i in range(100):
    if probability_of_error(data_set, i / 100) < probability_of_error(data_set, threshold):
        threshold = i / 100
plt.plot(ROC_curve(data_set, threshold)[1], ROC_curve(data_set, threshold)[0], 'ro')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
