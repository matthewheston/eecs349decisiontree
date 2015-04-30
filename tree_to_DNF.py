import pickle
import argparse
from decision_tree import is_leaf, getAttr

def tree_to_disjunctive_normal(tree):
    '''Takes a tree, and returns a string representing
    the disjunctive normal representation of the tree'''
    conditions = []
    def find_conditions(tree, curr_string = '('):
        print curr_string
        if is_leaf(tree):
            if tree['label'] == 1:
                conditions.append(curr_string + ')')
        else:
            attr = getAttr(tree)
            for x in tree[attr]:
                # Create a new string to describe the next part of the
                # tree. If this is the start of a string, don't include
                # the AND
                curr_string += '{}{} = {}'.format((curr_string != '(') * ' AND ',
                        attr,x)
                find_conditions(tree[attr][x], curr_string)
    find_conditions(tree)
    return ' OR\n'.join(conditions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tree', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

with open(args.tree, 'rb') as t:
    tree = pickle.load(t)

with open(args.output, 'wb') as o:
    DNFString = tree_to_disjunctive_normal(tree)
    o.write(DNFString)
