import pickle
import argparse
from decision_tree import is_leaf, getAttr

def tree_to_disjunctive_normal(tree):
    '''
    Takes a tree, and returns a string representing
    the disjunctive normal representation of the tree and
    the number of splits in the tree.
    '''
    conditions = []
    def find_conditions(tree, curr_string = '(', count=0):
        if is_leaf(tree):
            if tree['label'] == 1:
                conditions.append(curr_string + ')')
        else:
            find_conditions.count += 1
            attr = getAttr(tree)
            for x in tree[attr]:
                # Create a new string to describe the next part of the
                # tree. If this is the start of a string, don't include
                # the AND
                prefix = '' if curr_string == '(' else ' AND '
                find_conditions(tree[attr][x], curr_string +
                        '{}{} = {}'.format(prefix, attr, x))
    find_conditions.count = 0
    find_conditions(tree)
    return (' OR\n'.join(conditions), find_conditions.count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--decision_tree', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-s', '--print_splits', action="store_true")

    args = parser.parse_args()

with open(args.decision_tree, 'rb') as t:
    tree = pickle.load(t)

with open(args.output, 'wb') as o:
    DNFString = tree_to_disjunctive_normal(tree)
    if args.print_splits:
        print DNFString[1]
    o.write(DNFString[0])
