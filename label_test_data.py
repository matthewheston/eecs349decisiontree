import pickle
import argparse
import pandas as pd
from decision_tree import predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--decision_tree', required=True)
    parser.add_argument('-t', '--test_data', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

with open(args.decision_tree, 'rb') as t:
    tree = pickle.load(t)

with open(args.test_data) as f:
    test_df = pd.read_csv(f)
    test_df.columns = map(lambda c: c.strip(), test_df.columns)

predict(tree, test_df, 'winner')
test_df.to_csv(args.output)
