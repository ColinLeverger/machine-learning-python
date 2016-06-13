"""
Created on Wed May 25 2016

@author: Colin LEVERGER, Anais GALISSON
"""
from code.tools import *
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# Do the nearest_neighbors benchmark (training, choice of best hyperparameter, test of learning)
def do_nearest_neighbors(instances_train, target_train, instances_test, target_test):
    print('')
    print("###############################")
    print("### Model Nearest Neighbors ###")
    print("###############################")

    score = {}
    for n in range(1, 50):
        # Define a nearest neighbour model
        decision_nearest_neighbors = KNeighborsClassifier(n_neighbors=n, algorithm='auto')

        act_score = get_score(decision_nearest_neighbors, instances_train, target_train)
        print("For n " + str(n) + " the scores are " + str(act_score))
        score[n] = act_score.mean()

    score = sorted(score.items(), key=lambda x: x[1])
    best_n_neighbors = score[-1][0]

    # The best number of neighbors for this dataset is:
    print("The best number of neighbors for this dataset is " + str(best_n_neighbors))

    decision_nearest_neighbors = KNeighborsClassifier(n_neighbors=best_n_neighbors, algorithm='auto')

    # Fit the model using just the test set
    decision_nearest_neighbors.fit(instances_train, target_train)

    # Use the model to make predictions for the test set queries
    predictions_nearest_neighbors = decision_nearest_neighbors.predict(instances_test)
    print('')
    print("# Accuracy #")
    print("Accuracy = " + str(accuracy_score(target_test, predictions_nearest_neighbors, normalize=True)))
    print('')

    print("# Confusion Matrix #")
    plot_confusion_matrix(target_test, predictions_nearest_neighbors)
