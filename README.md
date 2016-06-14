# Machine Learning with Python

Machine learning project, use of Python, Plotly and [Scikit-learn](http://scikit-learn.org/stable/) library; we learned how to build, use and benchmark machine learning algorithms/mathematical models.

# Quick overview

## Goal of this project

The goal of this project was to have an overview of machine learning. 

We had to build & test algorithms, going from DQR to complex Data Analitic. Because we are only computer scientist yet (and not statistician), we played a lot with functions and benchmarked algorithms, by using our feelings. That is why some of the algorithms used seems to be inappropriate considering the given dataset (Linear Lasso Regression for instance). But we understood better after coding the mathematical solutions.

## Arborecence of the project

### Data Quality Report (first dataset)

- "./data": data used to generate the DQR,
- "./outputs": outputs of algorithm:
    + plotly HTML files with bars/graphs/hists...
    + DQR in .csv format.
- "./screenshots": two screenshots of the plotly HTML graphs generated,
- "./DQR-V1.py" and "./DQR-V2.py": code used to generate DQR, two versions:
    + V1: with standards scikit library,
    + V2: coded from scratch!

For this first part, I worked with Valerian SALIOU (https://crisp.im).

### Modelling (second dataset)

- "./code": functions and code used in "./main.py"; generate models, tests, benchmarks, clean data...
- "./data": dataset used,
- "./dqr_outputs": outputs and graphs for DQR,
- "./main.py": code that will orchestrate the benchmarks of differents algorithms.

For this second part, I worked with Anais GALISSON.

# Focus on the project & used methods

## First part: generate the Data Quality Report!

The DQR-generation step is an important part of the Data analytic process; with a clean report, it is possible de have insights about the data, to detect problems (missing values, strange values, outliers, etc.) and to make decision about these problems.

For this first assignment, we worked with a first dataset; the goal was to predict the salary of a person, consideing his age, the job of his parents, his level of education, etc.

For this first task we didn't build any models; the goal was just to create & manipulate a DQR.

mettre les deux DQR en screenshots

## Second part: modelling & predictions

For this second part, we had to work on the models themselves; we had to choose three models and we had to test them with the data, to chose the most appropriate (and try to explain why it was the most appropriate).

### The steps followed for this part were:

1. First, use of the DQR code of first part to generate the DQR for the given dataset, and take decisions about problems detected in the set. In our case, the dataset was totally clean (no misses, no outliers, ... perfect!) and it was OK to use it as-it was.
2. Implement models,
3. Analyse confusion matrixes and accuracy results to chose the best algorithm,
4. Conclude: why this model was better than this other one?

We worked with a second dataset, which concerned the alcools and the grades in Highschool. The goal was to predict the grade of a student, considering his weekly alcoohol consumtion.

### Results of an execution...

mettre une matrix de confusion


Code output example:

'''
###########################
### Model Decision Tree ###
###########################
For criterion entropy the scores are [ 0.57575758  0.55384615  0.609375    0.67213115  0.68333333]
For criterion gini the scores are [ 0.54545455  0.69230769  0.609375    0.68852459  0.66666667]
The best hyper parameter in this case for the Model Decision Tree was gini

# Accuracy #
Accuracy = 0.582278481013

# Confusion Matrix #

###########################
### Model Random Forest ###
###########################
For criterion entropy the scores are [ 0.68181818  0.63076923  0.671875    0.67213115  0.68333333]
For criterion gini the scores are [ 0.72727273  0.67692308  0.640625    0.70491803  0.66666667]
The best hyper parameter in this case for the Model Random Forest was gini

# Accuracy #
Accuracy = 0.670886075949

# Confusion Matrix #

###############################
### Model Nearest Neighbors ###
###############################
For n 1 the scores are [ 0.57575758  0.56923077  0.5625      0.57377049  0.53333333]
For n 2 the scores are [ 0.48484848  0.53846154  0.5625      0.59016393  0.63333333]
...
For n 5 the scores are [ 0.60606061  0.53846154  0.640625    0.59016393  0.6       ]
For n 6 the scores are [ 0.65151515  0.56923077  0.625       0.60655738  0.56666667]
For n 7 the scores are [ 0.66666667  0.6         0.640625    0.60655738  0.56666667]
...
For n 46 the scores are [ 0.59090909  0.50769231  0.640625    0.57377049  0.66666667]
For n 47 the scores are [ 0.57575758  0.52307692  0.625       0.59016393  0.65      ]
For n 48 the scores are [ 0.57575758  0.47692308  0.578125    0.59016393  0.66666667]
For n 49 the scores are [ 0.57575758  0.47692308  0.578125    0.59016393  0.66666667]
The best number of neighbors for this dataset is 15

# Accuracy #
Accuracy = 0.632911392405

# Confusion Matrix #

###############################
### Model Linear Regression ###
###############################
Coefficients for LinearRegression:
 [ -1.12056428e-01   8.60336080e-03   9.44102565e-02  -8.56978851e-02
  -5.61255131e-02   1.58404580e-03   1.35585020e-02  -3.55624437e-02
  -1.31684645e-03   1.27718032e-01   2.47913987e-01  -9.13592399e-03
  -3.95554345e+10  -3.95554345e+10  -3.95554345e+10  -3.95554345e+10
  -3.95554345e+10   6.00245012e-02  -9.01743471e+09  -9.01743471e+09
  -9.01743471e+09  -9.01743471e+09  -9.01743471e+09  -7.36754287e+10
  -7.36754287e+10  -2.68648177e+11  -2.68648177e+11   1.16301526e+11
   1.16301526e+11   4.46974197e+10   4.46974197e+10   2.03854421e+11
   2.03854421e+11   5.02692427e+11   5.02692427e+11   5.02692427e+11
   7.89833795e+10   7.89833795e+10  -9.51988693e+09  -9.51988693e+09
   1.73048537e+10   1.73048537e+10  -8.01148755e+08  -8.01148755e+08
  -4.34514687e+10  -4.34514687e+10  -4.34514687e+10  -4.34514687e+10
   2.87836634e+11   2.87836634e+11  -3.19201118e+11  -3.19201118e+11
  -2.23312010e+11  -2.23312010e+11   1.83929953e+10   1.83929953e+10
   1.76526496e-01   8.54051430e-02]
Residual sum of squares for LinearRegression: 0.80
Coefficients for Lasso:
 [-0.         -0.          0.         -0.         -0.         -0.         -0.
 -0.         -0.          0.05923985  0.24852654  0.         -0.          0.
 -0.          0.         -0.          0.         -0.          0.         -0.
 -0.         -0.         -0.          0.         -0.          0.         -0.
  0.          0.         -0.          0.         -0.         -0.          0.
 -0.         -0.          0.         -0.          0.          0.         -0.
  0.         -0.         -0.         -0.         -0.          0.          0.
 -0.          0.         -0.          0.         -0.         -0.          0.
  0.         -0.        ]
Residual sum of squares for Lasso: 1.02


Variance score for LinearRegression: 0.71
R2 score for LinearRegression: 0.76
Variance score for Lasso: 0.64
R2 score for Lasso: 0.80

Process finished with exit code 0
'''

# Going further

If you are interested about my work, please read the reports we wrote about it, which are available in: lien rapport
