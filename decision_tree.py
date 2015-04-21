from __future__ import division
import pandas as pd
import argparse
import scipy.stats as stats
from pprint import pprint


def make_tree(df, class_column):
    """
    The primary function which creates a decision tree.
    `df` is a pandas dataframe.
    `class_column` is the name of the column name that
    contains our output.
    """
    if not len(df.index):
        return df[class_column].value_counts().idxmax()
    if len(df[class_column].unique()) == 1:
        return df[class_column].unique()[0]
    if len(df.columns) == 1:
        return df[class_column].value_counts().idxmax()

    # base cases covered, let's get to our recursive bit
    best_attribute = find_best_attribute(df, class_column)
    tree = {best_attribute:{}}

    for value in df[best_attribute].unique():
        subtree = make_tree(df[df[best_attribute] == value], class_column)

        tree[best_attribute][value] = subtree


    return tree


def find_best_attribute(df, class_column):
    attr_info_gain = {} # store info gain for each attr here
    for attr in df.columns:
        if attr != class_column:
            attr_info_gain[attr] = get_information_gain(pd.DataFrame({"attr": df[attr], "target": df[class_column]}))

    return sorted(attr_info_gain.items(), key=lambda d: d[1])[-1][0]


def get_information_gain(df):
    values = df["attr"].unique()
    entropy_values = []
    for value in values:
        target = df[df["attr"] == value]["target"]
        positive_cases = sum(target)
        negative_cases = len(target) - sum(target)
        entropy = stats.entropy([positive_cases, negative_cases], base=2)
        weighted_entropy = len(df[df["attr"] == value])/len(df["attr"]) * entropy
        entropy_values.append(weighted_entropy)

    return stats.entropy([sum(df["target"]), len(df["target"]) - sum(df["target"])], base=2) - sum(entropy_values)

if __name__ == "__main__":
    # set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data')

    args = parser.parse_args()

    # read our data into pandas dataframe
    data_table = pd.read_csv(args.data)
    tree = make_tree(data_table,"edible")
    pprint(tree)

