"""
Created on Wed May 25 2016

@author: Colin & Anais
@note: Based on http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
"""
from code.create_data import create_data
from code.linear_regression import *
from code.nearest_neighbors import *
from code.random_forest import *
from code.decision_tree import *
import warnings

# Hmm, it is not classy but cant get rid of a warning...
warnings.filterwarnings("ignore")

# Read & create data from provided path
target_labels, train_descriptive_features = create_data('data/student-mat.csv', display_debugs=False)

# Create training instances
instances_train, instances_test, target_train, target_test = create_instances(train_descriptive_features, target_labels)

# Do decision tree model
do_decision_tree(instances_train, target_train, instances_test, target_test)

# Do random forest model
do_random_forest(instances_train, target_train, instances_test, target_test)

# Do nearest neighbors model
do_nearest_neighbors(instances_train, target_train, instances_test, target_test)

# Do linear regression
answer_dic = do_linear_regression(instances_train, instances_test, target_train, target_test, target_labels,
                                  display_debugs=True)
