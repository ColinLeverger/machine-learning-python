"""
Created on Thu May 26 15:56:42 2016

Set of tools & functions to manage data and plot things.

@author: Anais GALISSON
"""

from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Function to create instances for tests.
def create_instances(train_descriptive_features, target_labels):
    # Split the data: 80% training : 20% test set, because we have a few number of data
    instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(
        train_descriptive_features, target_labels, test_size=0.2, random_state=0)
    return instances_train, instances_test, target_train, target_test


def plot_confusion_matrix(target_test, predictions):
    conf = confusion_matrix(target_test, predictions)
    # Draw the confusion matrix
    # Show confusion matrix in a separate window
    plt.matshow(conf)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def get_score(decision_tree_model_entropy, instances_train, target_train):
    # print("\n# Score #")
    # Run a 5 fold cross validation on this model using the full census data
    score = cross_validation.cross_val_score(decision_tree_model_entropy, instances_train, target_train,
                                             cv=5)
    # The cross validation function returns an accuracy score for each fold
    # print("Score by fold: " + str(score))
    # We can output the mean accuracy score and standard deviation as follows:
    return score


# Create the frequency distribution for a normal distribution with two much cardinality
# Here we will create 6 new cardinality
def frequency_distribution(target):
    repartition = target.value_counts(ascending=True)
    index_in_order = repartition.index
    index_in_order = sorted(index_in_order, key=int)

    # How many values we wants for each parts
    step = round(len(target) / 5)

    # 6 categories
    A = []
    B = []
    C = []
    D = []
    E = []
    number_of_values = 0

    for i in index_in_order:
        number_of_values = number_of_values + repartition[i]
        if number_of_values < step:
            A.append(i)
        elif number_of_values < 2 * step:
            B.append(i)
        elif number_of_values < 3 * step:
            C.append(i)
        elif number_of_values < 4 * step:
            D.append(i)
        elif number_of_values < 5 * step:
            E.append(i)

    list_of_repartition = np.array([A, B, C, D, E]).tolist()

    for i in range(0, len(target)):
        if target[i] in A:
            target.set_value(i, "1")
        elif target[i] in B:
            target.set_value(i, "2")
        elif target[i] in C:
            target.set_value(i, "3")
        elif target[i] in D:
            target.set_value(i, "4")
        elif target[i] in E:
            target.set_value(i, "5")

    return target, list_of_repartition
