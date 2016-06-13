"""
Created on Wed May 25 2016

@author: Colin LEVERGER, Anais GALISSON
"""
from sklearn import tree
from code.tools import *
from sklearn.metrics import accuracy_score


# Build a decision tree
def build_tree(c):
    # Define a decision tree model using entropy based information gain
    return tree.DecisionTreeClassifier(criterion=c)


# Do the decision tree benchmark (training, choice of best hyperparameter, test of learning)
def do_decision_tree(instances_train, target_train, instances_test, target_test):
    print('')
    print("###########################")
    print("### Model Decision Tree ###")
    print("###########################")

    # Available criterions for decision tree
    criterion = ['entropy', 'gini']

    score = []
    for c in criterion:
        decision_tree = build_tree(c)

        act_score = get_score(decision_tree, instances_train, target_train)
        score.append(act_score)
        print("For criterion " + str(c) + " the scores are " + str(act_score))

    if score[0].mean() > score[1].mean():
        best_criterion = 'entropy'
    else:
        best_criterion = 'gini'

    # Print better algorithm in this case
    print("The best hyper parameter in this case for the Model Decision Tree was " + best_criterion)

    # Build tree with best criterion
    decision_tree = build_tree(best_criterion)

    # Fit the model using just the test set
    decision_tree.fit(instances_train, target_train)

    # Use the model to make predictions for the test set queries
    predictions_tree_model = decision_tree.predict(instances_test)

    # Output the accuracy score of the model on the test set
    print('')
    print("# Accuracy #")
    print("Accuracy = " + str(accuracy_score(target_test, predictions_tree_model, normalize=True)))
    print('')

    # Output the confusion matrix on the test set
    print("# Confusion Matrix #")
    plot_confusion_matrix(target_test, predictions_tree_model)
