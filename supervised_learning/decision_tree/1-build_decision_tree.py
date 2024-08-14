#!/usr/bin/env python3
''' count nodes in the tree '''

import numpy as np


class Node:
    ''' class that create a node '''
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        ''' the function find the maximum of the depths '''
        if self.is_leaf:
            return self.depth
        else:
            max_depth_left = self.depth
            max_depth_right = self.depth
            if self.left_child is not None:
                max_depth_left = self.left_child.max_depth_below()
            if self.right_child is not None:
                max_depth_right = self.right_child.max_depth_below()
            return max(max_depth_left, max_depth_right, self.depth)


class Leaf(Node):
    ''' class that create a leaf '''
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        ''' return self depth '''
        return self.depth


class Decision_Tree():
    ''' class that create the Decision Tree '''
    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def count_nodes(self, only_leaves=False):
        ''' Function to count nodes in the tree '''
        def count_r(node):
            if node is None:
                return 0
            if only_leaves and node.is_leaf:
                return 1
            if not only_leaves:
                return 1 + count_r(node.left_child) + count_r(node.right_child)
            return count_r(node.left_child) + count_r(node.right_child)

        return count_r(self.root)

    def depth(self):
        ''' function that call max depth below in root '''
        return self.root.max_depth_below()
