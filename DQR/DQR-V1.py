# Colin LEVERGER, Valerian SALIOU
# Data Quality Report generation
# V1: with transpose & describe functions
import pandas
import plotly
import plotly.graph_objs as go

# Load raw data from CSV
raw_data = pandas.read_csv(filepath_or_buffer='./data/DataSet.csv', header=0, index_col=0)

# Initialise names of expected assessment files
continuous_feature_path = './data/leverger-saliou-DQR-ContinuousFeatures.csv'
categorical_feature_path = './data/leverger-saliou-DQR-CategoricalFeatures.csv'

"""
Continuous feature management
"""

# Find continuous values
continuous = raw_data.select_dtypes(include=['int_', 'float64'])

# Store the computed missing values
continuous_missing_features = {
    'Miss.': [],
    'Card': []
}

# Describe & transpose
desc_continuous = (continuous.describe()).transpose()

# Compute missing values...
features = continuous.columns.values
for f in features:
    # Saving continuous[f] in a list to compute easily
    continuous_list = continuous[f].tolist()
    # Computing & saving values on the fly...
    count = len(continuous[f])
    continuous_missing_features['Miss.'].append((continuous_list.count(' ?') / count) * 100)
    card = len(set(continuous_list))
    continuous_missing_features['Card'].append(card)
    # Create & save plots
    if card >= 10:
        plotly.offline.plot({
            "data": [
                go.Histogram(
                    x=continuous[f]
                )
            ],
            "layout": go.Layout(
                title="Histogram for feature \"" + f + "\" with cardinality >=10"
            )
        }, filename="./data/%s.html" % f)
    else:
        plotly.offline.plot({
            "data": [
                go.Bar(
                    x=continuous[f].value_counts().keys(),
                    y=continuous[f].value_counts().values
                )
            ],
            "layout": go.Layout(
                title="Bar plot for feature \"" + f + "\" with cardinality <10"
            )
        }, filename="./data/%s.html" % f)

# Add computed missing results
desc_continuous.insert(1, "% Miss.", continuous_missing_features['Miss.'])
desc_continuous.insert(2, "Card.", continuous_missing_features['Card'])
# Arrange results to fit given template
desc_continuous = desc_continuous[['count', '% Miss.', 'Card.', 'min', '25%', 'mean', '50%', '75%', 'max', 'std']]
desc_continuous.columns = ['Count', '% Miss.', 'Card.', 'Min', '1st Qrt.', 'Mean', 'Median', '3rd Qrt.', 'Max',
                           'Std. Dev.']
# Write result in csv file
desc_continuous.to_csv(continuous_feature_path)

"""
Categorical Features management
"""
# Find categories
categories = raw_data.select_dtypes(exclude=['int_', 'float64'])

# Describe & transpose
desc_categories = categories.describe().transpose()
# Save in dict (this functions is interesting, should have noticed it before)
desc_categories_dict = desc_categories.to_dict()

# Store the computed missing values
categorical_missing_features = {
    '% Miss.': [],
    'Mode %': [],
    '2d Mode': [],
    '2d Mode Freq.': [],
    '2d Mode %': []
}

# Compute missing values
features = categories.columns.values
for f in features:
    categories_list = categories[f].tolist()
    count = len(categories[f])
    first_mode_freq = desc_categories_dict['freq'][f]
    categorical_missing_features['% Miss.'].append((categories_list.count(' ?') / count) * 100)
    categorical_missing_features['Mode %'].append((float(first_mode_freq) / count) * 100)
    # TODO: find the second mode !!
    categorical_missing_features['2d Mode'].append(0)
    categorical_missing_features['2d Mode Freq.'].append(0)
    categorical_missing_features['2d Mode %'].append(0)
    # Create & save histograms for categories
    data = [
        go.Bar(
            x=categories[f].value_counts().keys(),
            y=categories[f].value_counts().values
        )
    ]
    layout = go.Layout(
        title="Bar plot for categorical feature \"" + f + "\""
    )
    figure = go.Figure(data=data, layout=layout)
    name = "./data/" + f + ".html"
    plotly.offline.plot(figure, filename=name)

# Arrange output
desc_categories.insert(1, "% Miss.", categorical_missing_features['% Miss.'])
desc_categories.insert(2, "Mode %", categorical_missing_features['Mode %'])
desc_categories = desc_categories[['count', '% Miss.', 'unique', 'top', 'freq', 'Mode %']]
desc_categories.columns = ['Count', '% Miss.', 'Card.', 'Mode', 'Mode Freq.', 'Mode %']

# Write result in csv file
desc_categories.to_csv(categorical_feature_path)
