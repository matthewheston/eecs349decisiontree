import pickle
import argparse
import pandas as pd
from decision_tree import predict, preprocess_dataframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--decision_tree', required=True)
    parser.add_argument('-t', '--test_data', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

with open(args.decision_tree, 'rb') as t:
    tree = pickle.load(t)

with open(args.test_data) as f:
    test_df = pd.read_csv(f, na_values=["?"])
    test_df = preprocess_dataframe(test_df, handle_continuous=False)

predict(tree, test_df, 'winner')
test_df.to_csv(args.output)
