from __future__ import division
import pandas as pd
import argparse
import scipy.stats as stats
from pprint import pprint
import pickle
import csv
import random
import matplotlib.pyplot as plt
import math

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
        return {'label':default}
    if len(df[class_column].unique()) == 1:
        return {'label':df[class_column].unique()[0]}
    if len(df.columns) == 1:
        return {'label':df[class_column].value_counts().idxmax()}

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

def preprocess_dataframe(df, metadata = '', class_column = '', handle_continuous=True):
    # remove whitespace from column names
    df.columns = map(lambda c: c.strip(), df.columns)
    if handle_continuous:
        df = handle_continuous_attributes(df, metadata, class_column)
    df = df.fillna(method='bfill')


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

def prune_tree(tree):
    '''
	Takes a labeled tree, created with the classify_tree function.
    Checks each of the terminal nodes of the tree to see if removing
    them will improve the performance on the validation set.
	'''
    if is_leaf(tree):
        return get_error(tree)
    else:
        attr = getAttr(tree)
        error = sum([prune_tree(tree[attr][x]) for x in tree[attr]])
        # If the error is the same or better from making this
        # a leaf, then make this a leaf instead
        if no_examples(tree):
            # If there aren't any examples of the current leaf,
            # then there isn't any error.
            return 0
        if error < min(tree['pos'], tree['total'] - tree['pos']):
            return error
        else:
            #print "Removing {}".format(attr)
            tree.pop(attr)
            if tree['pos'] > tree['total'] - tree['pos']:
                tree['label'] = 1
                return tree['total'] - tree['pos']
            else:
                tree['label'] = 0
                return tree['pos']

def no_examples(tree):
    return 'total' not in tree.keys()

def is_leaf(tree):
    return 'label' in tree.keys()

def get_error(tree):
    if 'total' not in tree:
        return 0
    if tree['label'] == 1:
        return tree['total'] - tree['pos']
    else:
        return tree['pos']

def getAttr(tree, ignoreList = ['total', 'pos']):
    '''
    Given a tree and a list of keys which are not attributes,
    finds and returns the attribute that is to be examined
    '''
    return [x for x in tree if x not in ignoreList][0]

def classify(tree, validationData, class_column):
    '''
    Takes a tree and validation data, modifies the tree
    to label it with counts for how often each class value
    appears in the validation set at each node.
    '''
    for i, row in validationData.iterrows():
        classify_instance(tree, row, class_column)
    return tree

def classify_instance(tree, instance, class_column):
    '''
    Takes a tree, a validation instance, and the class column.
    Modifies the tree in place to add the count for each
    class column value to each node.
    '''
    # Get the class label for the instance and add it to the
    # current node
    label = instance[class_column]
    try:
        tree['total'] +=1
        tree['pos'] += 1 * label
    except KeyError:
        tree['total'] = 1
        tree['pos'] = 1 * label
    # Check if this is a leaf node. If not, then traverse the tree
    try:
        tree['label']
    except KeyError:
        # Get attribute (it's whatever is left)
        attr = getAttr(tree)
        attrVal = test_ineq(instance, attr)
        # Try to get the next node down the tree.
        try:
            classify_instance(tree[attr][attrVal],
                    instance, class_column)
        # If the value isn't in the tree, there's nothing to do.
        except KeyError:
            pass

def predict(tree, data, predict_col = 'predicted'):
    '''
	Takes a decision tree, and a dataset, and adds a "predicted"
    column of predicted class labels.
	'''
    predictions = []
    # Created an empty column in the data frame
    if predict_col in list(data):
        print 'Column "{}" already existed in the data frame, and is being overwritten.'.format(predict_col)
    else:
        data[predict_col] = ''
    # Iterate through each item in the set, and get the classification
    for i, row in data.iterrows():
        predictions.append(predict_instance(tree, row))
    data['predicted'] = pd.Series(predictions)
    return data


def predict_instance(tree, row):
    # if it's a leaf node, return the label
    if 'label' in tree:
        return tree['label']
    else:
        # Get the key, which is the node name
        node = getAttr(tree)
        # Test the inequality with this data
        node_value = test_ineq(row, node)
        try:
            return predict_instance(tree[node][node_value], row)
        # If the node doesn't exist, that means the training set didn't have any examples.
        # Figure out how to choose this
        except KeyError:
            return choose_prediction(tree)


def choose_prediction(tree):
    return 1 if random.random() > .5 else 0


def test_ineq(row, expression):
    '''
	Given a row of data, and an expression, determines whether the
    expression is a category or is an inequality. If it's an inequality,
    it returns the evaluation of the inequality (True or False).
	'''
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
    '''
	Given two boolean lists/vectors/Series, computes the ratio
    of values which are the same in both vectors
	'''
    return sum(vec1 == vec2) / float(len(vec1))

def get_class_column(metadata_file, dataFrame):
    '''
	Given a metadata_file which includes a "class" designation,
    and a data frame. Returns the index of the class column
	'''
    dataFrame.columns = map(lambda c: c.strip(), dataFrame.columns)
    with open(metadata_file, 'rb') as f:
        reader = csv.reader(f)
        column_metadata = reader.next() 
        try:
            index = column_metadata.index('class')
            return list(dataFrame)[index]
        except ValueError:
            print '"class" not in metadata'


def make_learning_curve(training_df, validation_df, class_column, filename, training_sizes):
    accuracies = [] # store accuracies for each size here

    for size in training_sizes:
        training = training_df.head(size)
        tree = make_tree(training, class_column)
        labeled_tree = classify(tree, validation_df, class_column)
        predict(labeled_tree, validation_df)
        validation_accuracy = accuracy_score(validation_df[class_column],
                validation_df['predicted'])
        accuracies.append(validation_accuracy)

    # now that we have our Y (accuracies) and X (training_sizes), make plot
    plt.plot(training_sizes, accuracies)
    plt.xlabel("Training size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.savefig(filename)



if __name__ == "__main__":
    # set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-m', '--metadata', required=True)
    parser.add_argument('-v', '--validation_data')
    parser.add_argument('-p', '--prune', action='store_true')
    parser.add_argument('-o', '--output_pickle',
            default='./finished_tree.pkl')

    args = parser.parse_args()

    # read our data into pandas dataframe
    data_table = pd.read_csv(args.data, na_values=["?"])
    # Get the metadata, and preprocess
    class_column = get_class_column(args.metadata, data_table)
    data_table = preprocess_dataframe(data_table, args.metadata, class_column)
    print "Making decision tree..."
    tree = make_tree(data_table, class_column)
    # pprint(tree)
    if args.validation_data:
        validation_df = pd.read_csv(args.validation_data, na_values=["?"])
        validation_df = preprocess_dataframe(validation_df,
                args.metadata, class_column, False)
        labeled_tree = classify(tree, validation_df, class_column)
        if args.prune:
            print "Pruning tree..."
            error = prune_tree(labeled_tree)
        # Get validation data accuracy
        print "Classifying accuracy..."
        validation_df = predict(labeled_tree, validation_df)
        validation_accuracy = accuracy_score(validation_df[class_column],
                validation_df['predicted'])
        print "Accuracy: {}".format(validation_accuracy)
    else:
        # Change the name of "tree", so that it will output
        labeled_tree = tree

    with open(args.output_pickle, 'wb') as o:
        pickle.dump(labeled_tree, o)
