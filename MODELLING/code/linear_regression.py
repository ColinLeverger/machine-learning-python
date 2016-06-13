"""
Created on Wed May 25 18:38:09 2016

@author: Colin LEVERGER
@note: Based on http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
"""
from sklearn import linear_model
import sklearn
import numpy as np


# Function to do the linear regression benchmark
def do_linear_regression(instances_train, instances_test, target_train, target_test, target_labels, display_debugs):
    print('')
    print("###############################")
    print("### Model Linear Regression ###")
    print("###############################")
    # Create linear regression objects
    linear_regression = linear_model.LinearRegression()
    lasso_regression = linear_model.Lasso()

    # Train the models using the training sets
    linear_regression.fit(instances_train, target_train)
    lasso_regression.fit(instances_train, target_train)

    if display_debugs:
        # The coefficients
        print('Coefficients for LinearRegression: \n', linear_regression.coef_)
        # The mean square error
        print("Residual sum of squares for LinearRegression: %.2f"
              % np.mean((linear_regression.predict(instances_train) - target_train) ** 2))

        # The coefficients
        print('Coefficients for Lasso: \n', lasso_regression.coef_)
        # The mean square error
        print("Residual sum of squares for Lasso: %.2f"
              % np.mean((lasso_regression.predict(instances_train) - target_train) ** 2))

        print('')

    # Explained variance score: 1 is perfect prediction
    # Variance is the accuracy of the dataset for the training data
    # R2 is the accuracy of the dataset for the test data
    # It could be useful to compare them --> this function will return these two values
    variance1 = linear_regression.score(instances_train, target_train)
    r2_regr = sklearn.metrics.r2_score(target_test, (linear_regression.predict(instances_test)))

    variance2 = lasso_regression.score(instances_train, target_train)
    r2_regr1 = sklearn.metrics.r2_score(target_test, (lasso_regression.predict(instances_test)))

    # Print debugs
    if display_debugs:
        print('')
        print('Variance score for LinearRegression: %.2f' % variance1)
        print('R2 score for LinearRegression: %.2f' % r2_regr)
        print('Variance score for Lasso: %.2f' % variance2)
        print('R2 score for Lasso: %.2f' % r2_regr1)

    # Return results of experimentation in a dic format
    return {
        'Variance1': variance1,
        'R2 score 1': r2_regr,
        'Variance2': variance2,
        'R2 score 2': r2_regr1
    }
