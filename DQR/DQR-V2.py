# Valerian SALIOU
# Data Quality Report generation
# V2: from scratch!
import pandas, numpy, operator, copy

from plotly.offline import plot
from plotly.graph_objs import Bar, Histogram


class DQR:
    """
    Handles DQR operations
    """

    def __init__(self, data_path):
        assert data_path

        self.__data = pandas.read_csv(data_path)
        self.__with_graphs = True

    def generate_continuous(self):
        """
        Generates continuous features
        """
        print("Generating continuous report...")
        print()

        columns, statistics = self.__initialize_statistics_continuous()

        for feature in self.__data:
            feature_values = []

            # Parse feature values (pick the continuous ones)
            for feature_value in self.__data[feature]:
                feature_value = ("%s" % feature_value).strip()

                # Skip non-numeric features
                if feature_value.isnumeric():
                    feature_values.append(int(feature_value))

            # Does this feature has continuous properties?
            if len(feature_values) > 0:
                # Process descriptive statistics
                self.__process_statistics_continuous(statistics, feature, feature_values)

                if self.__with_graphs:
                    # Render graph for this feature
                    if statistics['Card.'][len(statistics['Card.']) - 1] >= 10:
                        # High cardinality
                        plot(
                            [Histogram(
                                x=self.__data[feature]
                            )],

                            filename=("./data/%s.html" % feature)
                        )
                    else:
                        # Low cardinality
                        occurences = self.__count_occurences(self.__data[feature])

                        plot([Bar(
                            x=[key for key, value in occurences],
                            y=[value for key, value in occurences]
                        )], filename=("./data/%s.html" % feature))

        df = pandas.DataFrame(statistics)
        df = df.reindex(columns=columns)

        df.to_csv("./data/leverger-saliou-DQR-ContinuousFeatures.csv")

        print(df)

        print()
        print("Done generating continuous report.")

    def generate_categorical(self):
        """
        Generates categorical features
        """
        print("Generating categorical report...")
        print()

        columns, statistics = self.__initialize_statistics_categorical()

        for feature in self.__data:
            # Skip ID
            if feature == "id":
                continue

            feature_values = []

            # Parse feature values (pick the categorical ones)
            for feature_value in self.__data[feature]:
                feature_value = ("%s" % feature_value).strip()

                # Skip numeric features
                if not feature_value.isnumeric() and feature_value != "?":
                    feature_values.append(feature_value)

            # Does this feature has categorical properties?
            if len(feature_values) > 0:
                # Process descriptive statistics
                self.__process_statistics_categorical(statistics, feature, feature_values)

                # Render graph for this feature
                if self.__with_graphs:
                    occurences = self.__count_occurences(feature_values)

                    plot([Bar(
                        x=[key for key, value in iter(occurences.items())],
                        y=[value for key, value in iter(occurences.items())]
                    )], filename=("./data/%s.html" % feature))

        df = pandas.DataFrame(statistics)
        df = df.reindex(columns=columns)

        df.to_csv("./data/leverger-saliou-DQR-CategoricalFeatures.csv")

        print(df)

        print()
        print("Done generating categorical report.")

    def __initialize_statistics_continuous(self):
        columns = [
            'Feature',
            'Count',
            '% Miss.',
            'Card.',
            'Min',
            '1st Qrt.',
            'Mean',
            'Median',
            '3rd Qrt.',
            'Max',
            'Std. Dev.'
        ]

        statistics = {
            'Feature': [],
            'Count': [],
            '% Miss.': [],
            'Card.': [],
            'Min': [],
            '1st Qrt.': [],
            'Mean': [],
            'Median': [],
            '3rd Qrt.': [],
            'Max': [],
            'Std. Dev.': []
        }

        return columns, statistics

    def __initialize_statistics_categorical(self):
        columns = [
            'Feature',
            'Count',
            '% Miss.',
            'Card.',
            'Mode',
            'Mode Count',
            'Mode %',
            '2nd Mode',
            '2nd Mode Count',
            '2nd Mode %'
        ]

        statistics = {
            'Feature': [],
            'Count': [],
            '% Miss.': [],
            'Card.': [],
            'Mode': [],
            'Mode Count': [],
            'Mode %': [],
            '2nd Mode': [],
            '2nd Mode Count': [],
            '2nd Mode %': []
        }

        return columns, statistics

    def __process_statistics_continuous(self, statistics, feature, feature_values):
        """
        Processes statistics on given feature values
        """
        statistics['Feature'].append(feature)
        statistics['Count'].append(len(feature_values))
        statistics['% Miss.'].append((1 - (len(feature_values) / len(self.__data[feature]))) * 100)
        statistics['Card.'].append(len(set(feature_values)))
        statistics['Min'].append(numpy.min(feature_values))
        statistics['1st Qrt.'].append(numpy.percentile(feature_values, 25))
        statistics['Mean'].append(numpy.mean(feature_values))
        statistics['Median'].append(numpy.median(feature_values))
        statistics['3rd Qrt.'].append(numpy.percentile(feature_values, 75))
        statistics['Max'].append(numpy.max(feature_values))
        statistics['Std. Dev.'].append(numpy.std(feature_values))

    def __process_statistics_categorical(self, statistics, feature, feature_values):
        """
        Processes statistics on given feature values
        """
        mode_occurences = self.__count_occurences(feature_values)

        # Process mode
        max_mode_1st = max(iter(mode_occurences.items()), key=operator.itemgetter(1))[0]

        mode_occurences_without_1st = copy.copy(mode_occurences)
        mode_occurences_without_1st.pop(max_mode_1st, None)

        max_mode_2nd = max(iter(mode_occurences_without_1st.items()), key=operator.itemgetter(1))[0]

        # Push statistics
        statistics['Feature'].append(feature)
        statistics['Count'].append(len(feature_values))
        statistics['% Miss.'].append((1 - (len(feature_values) / len(self.__data[feature]))) * 100)
        statistics['Card.'].append(len(set(feature_values)))
        statistics['Mode'].append(max_mode_1st)
        statistics['Mode Count'].append(mode_occurences[max_mode_1st])
        statistics['Mode %'].append((mode_occurences[max_mode_1st] / len(feature_values)) * 100)
        statistics['2nd Mode'].append(max_mode_2nd)
        statistics['2nd Mode Count'].append(mode_occurences[max_mode_2nd])
        statistics['2nd Mode %'].append((mode_occurences[max_mode_2nd] / len(feature_values)) * 100)

    def __count_occurences(self, values):
        """
        Utility to count occurences
        """
        occurences = {}

        for value in values:
            value = value.strip()

            if not value in occurences:
                occurences[value] = 0

            occurences[value] += 1

        return occurences


# Proceed
dqr = DQR("./data/DataSet.csv")

dqr.generate_continuous()
dqr.generate_categorical()
