from __future__ import division
import pandas as pd
import argparse
import scipy.stats as stats
from pprint import pprint
import pickle

import sys
# sys.setrecursionlimit(10000)


def make_tree(df, class_column, default=0):
    """
    The primary function which creates a decision tree.
    `df` is a pandas dataframe.
    `class_column` is the name of the column name that
    contains our output.
    """
    if not len(df.index):
        return default
    if len(df[class_column].unique()) == 1:
        return df[class_column].unique()[0]
    if len(df.columns) == 1:
        return df[class_column].value_counts().idxmax()

    # base cases covered, let's get to our recursive bit
    best_attribute = find_best_attribute(df, class_column)
    tree = {best_attribute:{}}

    for value in df[best_attribute].unique():
        subset_data = df[df[best_attribute] == value]
        subset_data = subset_data.drop(best_attribute, axis=1)
        subtree = make_tree(subset_data, class_column)

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

def preprocess_dataframe(df, class_column):
    # remove whitespace from column names
    df.columns = map(lambda c: c.strip(), df.columns)
    df = handle_continuous_attributes(df, class_column)
    df = df.fillna(method="ffill")

    return df

def handle_continuous_attributes(df, class_column):
    # first, find continuous attributes
    continuous_columns = [c for c in df.columns if "-NUM" in c and not ">" in c]

    # now discretize them
    for column in continuous_columns:
        best_split = find_split_points(pd.DataFrame({"attr": df[column], class_column: df[class_column]}), column, class_column)
        df[best_split[0]] = best_split[1]

    for column in continuous_columns:
        df = df.drop(column, axis=1)

    return df


def find_split_points(df, column_name, class_column):
    positive_avg = df[df[class_column] == 1]["attr"].mean()
    negative_avg = df[df[class_column] == 0]["attr"].mean()
    midway = (positive_avg + negative_avg) / 2
    column_name = "%s > %.5f" % (column_name, midway)
    discretized = df.apply(lambda row: str(row["attr"] > midway), axis=1)
    df[column_name] = discretized
    return (column_name, df[column_name])
    # split_candidates = []
    # last_row = None
    # # get candidates
    # for index, row in df.iterrows():
    #     if last_row is not None and two_rows_ago is not None:
    #         if row[class_column] != last_row[class_column]:
    #             split_candidates.append((row["attr"] + last_row["attr"]) / 2)
    #     last_row = row

    # split_candidates = pd.Series(split_candidates)
    # print split_candidates.dropna()
    # for candidate in split_candidates:
    #     column_name = "%s > %.6f" % (column_name, candidate)
    #     # print column_name
    #     discretized = df.apply(lambda row: str(row["attr"] > candidate), axis=1)
    #     df[column_name] = discretized

    # best_split = find_best_attribute(df.drop("attr", axis=1), class_column)

    # return (best_split, df[best_split])

def prune_tree(tree, validation_data):
    '''Takes a tree (in the form of a dictionary of dictionaries),
    and validation data (in the form of a pandas data frame).
    Checks each of the terminal nodes of the tree to see if removing
    them will improve the performance on the validation set.'''
    validation_data = classify(tree, validation_data)

def classify(tree, data):
	'''Takes a decision tree, and a dataset, and adds a "predicted"
    column of predicted class labels.'''
    # Created an empty column in the data frame
    if not data['predicted']:
        data['predicted'] = np.nan
    else:
        raise KeyError('column "predicted" already exists in the data frame')
    # Iterate through each item in the set, and get the classification
    for i, row in data.iterrows():
        data['predicted'][i] = classify_instance(tree, row)

    return data

def testIneq(row, expression):
    '''Given a row of data, and an expression, determines whether the
    expression is a category or is an inequality. If it's an inequality,
    it returns the evaluation of the inequality (True or False).'''
    # Split the expression
    exprSplit = expression.split(' ')
    # If the expression has 3 parts, it's a numeric; Otherwise, it's categorical, and we just return the value
    if len(exprSplit) == 1:
        return row[expression]
    else:
        try:
            column, expr, value = exprSplit
        except:
            print "Poorly formed expression: {}".format(expression)
            return None
        if expr == "<=":
            return row[column] <= float(value)
        if expr == ">":
            return row[column] > float(value)
        else:
            raise ValueError('Malformed inequality: {}'.format(expression))


if __name__ == "__main__":
    # set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data')
    parser.add_argument('-p', '--prune', action='store_true')
    parser.add_argument('-v', '--validation_data')

    args = parser.parse_args()

    # read our data into pandas dataframe
    data_table = pd.read_csv(args.data, na_values=["?"])
    data_table = preprocess_dataframe(data_table, "winner")
    tree = make_tree(data_table,"winner")
    if args.prune:
        if args.validation_data:
            validation_data_table = pd.read_csv(args.validation_data)
            tree = prune_tree(tree, validation_data_table)
    with open('finished_tree.pkl', 'wb') as o:
        pickle.dump(tree, o)
    pprint(tree)

