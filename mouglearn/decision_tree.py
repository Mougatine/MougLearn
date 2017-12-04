import math
import itertools
import operator as op

import numpy as np
import pandas as pd
import ete3

class Tree:
    def __init__(self, attr, val, root=False, ete3_node=None):
        self.attr = attr
        self.val = val
        self.root = root
        if root:
            self.ete3_node = ete3.Tree(name='Root')
        else:
            self.ete3_node = ete3_node

        self.children = []


    def add_child(self, attr, val, df=None, target=None, continuous=False):
        if attr is None:
            if not continuous:
                distribution = round(len(df[df[target] == val]) / len(df), 2) * 100
                msg = ' {}: {}% '.format(val, int(distribution))
            else:
                msg = ' {}: {} '.format(target, round(val, 2))
        elif not continuous:
            msg = ' {} = {} '.format(attr, val)
        else:
            msg = ' {} {} {} '.format(attr, '{}', round(val, 2))

        if len(self.children) == 0:
            ete3_child =  self.ete3_node.add_child(name=msg)
        else:
            ete3_child = self.children[-1].ete3_node.add_sister()

        if attr is not None and not continuous:
            ete3_child.add_face(ete3.TextFace(msg),
                                column=0, position='branch-right')
        elif attr is not None and len(self.children) == 0:
            ete3_child.add_face(ete3.TextFace(msg.format('<')),
                                column=0, position='branch-right')
        elif attr is not None:
            ete3_child.add_face(ete3.TextFace(msg.format('>=')),
                                column=0, position='branch-right')

        node = Tree(attr, val, ete3_node=ete3_child)
        self.children.append(node)

        return node


class ID3:
    def __init__(self, criterion='entropy'):
        self.criterion = criterion

        if self.criterion == 'entropy':
            self.compute_metric = self.compute_entropy
        elif self.criterion == 'infogain':
            self.compute_metric = self.compute_info_gain
        elif self.criterion == 'gainratio':
            self.compute_metric = self.compute_gain_ratio
        elif self.criterion == 'gini':
            self.compute_metric = self.compute_gini
        else:
            raise ValueError('Unknown criterion {}.'.format(self.criterion))


    def fit(self, df, target_attr, attrs, criterion='entropy'):
        self.df = df
        self.root = Tree(None, None, root=True)
        self.target = target_attr
        self.build_tree(self.root, df, target_attr, attrs)


    def build_tree(self, node, df, target_attr, attrs):
        if len(df) == 0:
            continuous = df[target_attr].dtype != np.dtype('O')
            node.add_child(None, self.df[target_attr].value_counts().max(),
                           df=self.df, target=target_attr, continuous=continuous)
            return
        elif len(attrs) == 0 or df[target_attr].value_counts().max() == len(df): # Leaf
            if df[target_attr].dtype == np.dtype('O'): # Classification task
                node.add_child(None, df[target_attr].value_counts().idxmax(),
                               df=df, target=target_attr)
            else: # Regression task
                node.add_child(None, df[target_attr].mean(), df=df,
                               target=target_attr, continuous=True)
            return

        best_attr_idx = self.get_best_attr(df, target_attr, attrs)
        best_attr = attrs[best_attr_idx]
        remaining_attrs = attrs[:best_attr_idx] + attrs[best_attr_idx+1:]

        if df[best_attr].dtype == np.dtype('O'): # Categorical values
            for val in df[best_attr].unique():
                filtered_examples = df[df[best_attr] == val]
                child = node.add_child(best_attr, val)
                self.build_tree(child, filtered_examples, target_attr, remaining_attrs)
        else: # Continuous values
            val = df[best_attr].median()
            filtered_examples = df[df[best_attr] < val]
            child = node.add_child(best_attr, val, continuous=True)
            self.build_tree(child, filtered_examples, target_attr, remaining_attrs)

            filtered_examples = df[df[best_attr] >= val]
            child = node.add_child(best_attr, val, continuous=True)
            self.build_tree(child, filtered_examples, target_attr, remaining_attrs)



    def predict(self, df_example):
        node = self.root
        while True:
            if len(node.children) == 1 and len(node.children[0].children) == 0:
                return node.children[0].val

            attr = node.children[0].attr
            if isinstance(df_example[attr], str): # Categorical value
                for child in node.children:
                    if df_example[attr] == child.val:
                        node = child
                        break
            else: # Continuous value
                if df_example[attr] < node.children[0].val:
                    node = node.children[0]
                else:
                    node = node.children[1]


    def display_tree(self):
        ts = ete3.TreeStyle()
        ts.show_scale = False
        ts.show_leaf_name = True
        return self.root.ete3_node.render('%%inline', w=200, units='mm', tree_style=ts)


    def score(self, df):
        nb_elts = len(df)
        good_preds = 0
        for i in range(nb_elts):
            pred, true = self.predict(df.iloc[i]), df.iloc[i][self.target]
            if pred == true:
                good_preds += 1

        return good_preds / nb_elts


    def get_best_attr(self, df, target_attr, attrs):
        if self.criterion in ['entropy', 'gini']:
            val = float('inf')
            cmp = op.lt
        else:
            val = -float('inf')
            cmp = op.gt
        best_attr_idx = 0

        nb_elts = len(df)
        for attr_idx, attr in enumerate(attrs):
            new_val = self.compute_metric(df, nb_elts, target_attr, attr)
            if cmp(new_val, val):
                best_attr_idx = attr_idx
                val = new_val

        return best_attr_idx


    def compute_entropy(self, df, nb_elts, target_attr, attr):
        unique_vals = df[attr].unique()
        entropy = 0
        for x in unique_vals:
            p = len(df[df[attr] == x]) / nb_elts
            entropy += -p * math.log2(p)
        return entropy


    def compute_info_gain(self, df, nb_elts, target_attr, attr):
        attr_vals = list(df[attr].unique())
        target_vals = list(df[target_attr].unique())
        info_gain = 0

        for (attr_val, target_val) in itertools.product(attr_vals, target_vals):
            pac = len(df[(df[attr] == attr_val) & (df[target_attr] == target_val)]) + 1e-8
            pa = len(df[df[attr] == attr_val])
            pc = len(df[df[target_attr] == target_val])

            info_gain += pac * math.log2(pac / (pa * pc))

        return info_gain


    def compute_gain_ratio(self, df, nb_elts, target_attr, attr):
        info_gain = self.compute_info_gain(df, nb_elts, target_attr, attr)
        split_info = 0

        for attr_val in list(df[attr].unique()):
            si = len(df[df[attr] == attr_val])
            ratio = si / nb_elts
            split_info += ratio * math.log2(ratio)

        return -split_info


    def compute_gini(self, df, nb_elts, target_attr, attr):
        gini_impurity = 0

        if df[attr].dtype == np.dtype('O'): # Categorical attribute
            for attr_val in list(df[attr].unique()):
                p = len(df[df[attr] == attr_val]) / nb_elts
                gini_impurity += p ** 2
        else: # Continuous attribute
            # Split the data by its median as it is a better separator than
            # its mean.
            median = df[attr].median()
            p = len(df[df[attr] < median]) / nb_elts
            gini_impurity += p

            p = len(df[df[attr] >= median]) / nb_elts
            gini_impurity += p

        return 1 - gini_impurity
