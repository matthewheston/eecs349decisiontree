import csv
import pandas as pd
from decision_tree import *

TRAIN_DATA = './tests/XOR_train.csv'
VALIDATE_DATA = './tests/XOR_validate.csv'
METADATA = './tests/XOR_metadata.csv'


if __name__ == '__main__':
    # read our data into pandas dataframe
    data_table = pd.read_csv(TRAIN_DATA, na_values=["?"])
    # Get the metadata, and preprocess
    class_column = get_class_column(METADATA, data_table)
    data_table = preprocess_dataframe(data_table, METADATA, class_column)

    # Test to make sure that make_tree is working
    tree = make_tree(data_table, class_column)
    assert tree == {'V2': {1: {'V1': {1: {'label':0}, 0:{'label':1}}}, 0:{'V1':{1:{'label':1},0:{'label':0}}}}}
    
    # Test labeling the data
    validation_df = pd.read_csv(VALIDATE_DATA, na_values=["?"])
    validation_df = preprocess_dataframe(validation_df, METADATA, class_column, False)
    labeled_tree = classify(tree, validation_df, class_column)
    print labeled_tree
    assert labeled_tree == {'total':5, 'pos':4,'V2': {1: {'total': 2, 'pos':2, 'V1': {1:
        {'label':0, 'total':1, 'pos':1},
        0:{'label':1,'total':1,'pos':1}}},
        0:{'total':3, 'pos':2, 'V1':{1:
            {'label':1, 'total':2, 'pos':2},
            0:{'label':0, 'total':1, 'pos':0}}}}}

    # Test pruning the tree
    error = prune_tree(labeled_tree)
    assert labeled_tree == {'total':5, 'pos':4,'V2': {1: {'total': 2, 'pos':2, 'label':1},
        0:{'total':3, 'pos':2, 'V1':{1:
            {'label':1, 'total':2, 'pos':2},
            0:{'label':0, 'total':1, 'pos':0}}}}}
    
    # Test validation data accuracy
    predict(labeled_tree, validation_df)
    print validation_df
    validation_accuracy = accuracy_score(validation_df[class_column],
            validation_df['predicted'])
    print validation_accuracy
    assert validation_accuracy == 1
