"""
Created on Wed May 25 2016

@author: Colin LEVERGER, Anais GALISSON
"""
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from code.tools import frequency_distribution


# Function to create data to feed the algorithms
# 1) Data cleaning
# 2) Data enhancements
# Possible to prints debuts with "display_debugs" vars
def create_data(path, display_debugs):
    df = pd.read_csv(path, delimiter=';')

    # Extract Target Features
    target_labels = df['G3']
    target_labels, list_of_repartition = frequency_distribution(target_labels)

    # Extract Numeric Descriptive Features
    # We just keep continuous data
    df_continuous = df.select_dtypes(include=['int'])
    df_continuous = df_continuous.drop(['G3'] + ['traveltime'] + ['studytime'] + ['Medu'] + ['Fedu'], axis=1)

    # We get columns of continuous data in the table numeric_features
    numeric_descriptive_features = df_continuous

    # Extract Categorical Descriptive Features

    numeric_features = []
    for col in df_continuous.columns:
        numeric_features.append(col)

    categorical_descriptive_features = df.drop(numeric_features + ['G3'], axis=1)

    # Transpose into array of dictionaries of "feature:level" pairs
    categorical_descriptive_features = categorical_descriptive_features.T.to_dict().values()
    # Convert to numeric encoding
    vectorizer = DictVectorizer(sparse=False)
    vec_categorical_descriptive_features = vectorizer.fit_transform(categorical_descriptive_features)

    encoding_dictionary = vectorizer.vocabulary_

    if display_debugs:
        print("### Frequency Distribution ###")
        print("\n# List of the new repartition of the target #")
        print("First part  ,  Second Part , Third Part, ....")
        print(list_of_repartition)
        print("\n# Target Labels #")
        print(target_labels.head())
        print("\n### Numeric Descriptive Features ###")
        print(numeric_descriptive_features.head())
        print("\n\n ### Categorical Descriptive Features ### \n")
        print(categorical_descriptive_features.head())
        print("\n\n ### Output the numeric encoding mapping ###")
        for k in sorted(encoding_dictionary.keys()):
            mapping = k + " : column " + str(encoding_dictionary[k]) + " = 1"
            print(mapping)
        print("Processed Categorical features for first 5 instances:")
        print(vec_categorical_descriptive_features[0])
        print(vec_categorical_descriptive_features[1])
        print(vec_categorical_descriptive_features[2])
        print(vec_categorical_descriptive_features[3])
        print(vec_categorical_descriptive_features[4])

    # Merge Categorical and Numeric Descriptive Features
    return target_labels, np.hstack(
        (numeric_descriptive_features.as_matrix(), vec_categorical_descriptive_features))
