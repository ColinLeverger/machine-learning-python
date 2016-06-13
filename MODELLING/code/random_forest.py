"""
Created on Wed May 25 2016

@author: Colin LEVERGER, Anais GALISSON
"""
from sklearn import ensemble
from code.tools import *
from sklearn.metrics import accuracy_score

# Generic function to build a forest
def build_forest(c, n):
    return ensemble.RandomForestClassifier(criterion=c, n_estimators=n)

# Do the random forest benchmark
def do_random_forest(instances_train, target_train, instances_test, target_test):
    print('')
    print("###########################")
    print("### Model Random Forest ###")
    print("###########################")

    criterion = ['entropy', 'gini']
    # The random forest has good results with n = 100 !
    n = 100

    score = []
    for c in criterion:
        decision_random_forest_model = build_forest(c, n)

        act_score = get_score(decision_random_forest_model, instances_train, target_train)
        # Scores
        score.append(act_score)
        print("For criterion " + str(c) + " the scores are " + str(act_score))

    if score[0].mean() > score[1].mean():
        best_criterion = 'entropy'
    else:
        best_criterion = 'gini'

    # Print better algorithm in this case
    print("The best hyper parameter in this case for the Model Random Forest was " + best_criterion)

    decision_random_forest_model = build_forest(best_criterion, n)

    # Training
    decision_random_forest_model.fit(instances_train, target_train)

    # Predict
    predictions_random_forest = decision_random_forest_model.predict(instances_test)

    print('')
    print("# Accuracy #")
    print("Accuracy = " + str(accuracy_score(target_test, predictions_random_forest, normalize=True)))
    print('')

    print("# Confusion Matrix #")
    # Plots for confusion matrix
    plot_confusion_matrix(target_test, predictions_random_forest)
