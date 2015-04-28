from __future__ import division
import pandas as pd
import argparse
import scipy.stats as stats
from pprint import pprint
import pickle
import csv

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

def preprocess_dataframe(df, metadata, class_column, handle_continuous=True):
    # remove whitespace from column names
    df.columns = map(lambda c: c.strip(), df.columns)
    if handle_continuous:
        df = handle_continuous_attributes(df, metadata, class_column)
    df = df.fillna(method="bfill")

    return df


def handle_continuous_attributes(df, metadata_file, class_column):
    # first, find continuous attributes
    with open(metadata_file, 'rb') as f:
        reader = csv.reader(f)
        column_metadata = reader.next()
    continuous_columns = [df.columns[i] for i in range(len(df.columns)) if
                          column_metadata[i] == 'numeric']
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


def prune_tree(labeled_tree):
    '''Takes a tree (in the form of a dictionary of dictionaries),
    and validation data (in the form of a pandas data frame).
    Checks each of the terminal nodes of the tree to see if removing
    them will improve the performance on the validation set.'''
    if is_leaf(tree):
        return error(tree)
    else:
        error = sum([prune_tree(labeled_tree[x]) for x in labeled_tree])
        if error < 

def classify(tree, validationData):
    '''Takes a tree and validation data, returns a labeled
    tree, with counts for how often each 

def predict(tree, data):
    '''Takes a decision tree, and a dataset, and adds a "predicted"
    column of predicted class labels.'''
    # Created an empty column in the data frame
    if 'predicted' in list(data):
        raise KeyError('column "predicted" already exists in the data frame. Please rename this column')
    else:
        data['predicted'] = ''
    # Iterate through each item in the set, and get the classification
    for i, row in data.iterrows():
        data['predicted'][i] = predict_instance(tree, row)
    return None


def predict_instance(tree, row):
    label = None
    # Figure out if this is a leaf node. If so, return the node value
    # print tree.values()
    try:
        # Get the key, which is the node name
        node = tree.keys()[0]
        print node
        # Test the inequality with this data
        node_value = test_ineq(row, node)
        print node_value
        return predict_instance(tree[node][node_value], row)
    # If the subtree doesn't exist, then we're at a node
    except AttributeError:
        return tree


def test_ineq(row, expression):
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
            return str(row[column] <= float(value))
        if expr == ">":
            return str(row[column] > float(value))
        else:
            raise ValueError('Malformed inequality: {}'.format(expression))

def accuracy_score(vec1, vec2):
    '''Given two boolean lists/vectors/Series, computes the ratio of values which are
    the same in both vectors'''
    return float(sum(vec1 * vec2)) / len(vec1)

def get_class_column(metadata_file, dataFrame):
    '''Given a metadata_file which includes a "class" designation,
    and a data frame. Returns the index of the class column'''
    with open(metadata_file, 'rb') as f:
        reader = csv.reader(f)
        column_metadata = reader.next() 
        try:
            index = column_metadata.index('class')
            return list(dataFrame)[index]
        except ValueError:
            print '"class" not in metadata'


if __name__ == "__main__":
    # set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-m', '--metadata', required=True)
    parser.add_argument('-v', '--validation_data')
    parser.add_argument('-p', '--prune', action='store_true')

    args = parser.parse_args()

    # read our data into pandas dataframe
    data_table = pd.read_csv(args.data, na_values=["?"])
    # Get the metadata, and preprocess
    class_column = get_class_column(args.metadata, data_table)
    data_table = preprocess_dataframe(data_table, args.metadata, class_column)
    print "Making decision tree..."
    tree = make_tree(data_table, class_column)
    pprint(tree)
    if args.validation_data:
        validation_df = pd.read_csv(args.validation_data, na_values=["?"])
        validation_df.columns = map(lambda c: c.strip(), validation_df.columns)
        validation_df = preprocess_dataframe(validation_df, args.metadata, class_column, False)
        if args.prune:
            print "Pruning tree..."
            tree = prune_tree(tree, validation_df)
        # Get validation data accuracy
        print "Classifying accuracy..."
        classify(tree, validation_df)
        #print validation_df.head()
        validation_accuracy = accuracy_score(validation_df[class_column],
                validation_df['predicted'])
        print validation_accuracy
    with open('finished_tree.pkl', 'wb') as o:
        pickle.dump(tree, o)
