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

    def left_child_add_prefix(self, text):
        ''' adds a prefix to the left '''
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        ''' adds a prefix to the right '''
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        ''' creates the entire tree string '''
        current = "root" if self.is_root else "-> node"
        result = \
            f"{current} [feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            result += \
              self.left_child_add_prefix(str(self.left_child).strip())
        if self.right_child:
            result += \
              self.right_child_add_prefix(str(self.right_child).strip())
        return result

    def get_leaves_below(self):
        ''' function that obtains all the leaves '''
        return self.left_child.get_leaves_below()\
            + self.right_child.get_leaves_below()

    def update_bounds_below(self):
        ''' update bounds below '''
        if self.is_root:
            self.lower = {0: -1 * np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()

        if self.feature in self.left_child.lower.keys():
            self.left_child.lower[self.feature] = \
                max(self.threshold, self.left_child.lower[self.feature])
        else:
            self.left_child.lower[self.feature] = self.threshold

        if self.feature in self.right_child.upper.keys():
            self.right_child.upper[self.feature] = \
                min(self.threshold, self.right_child.upper[self.feature])
        else:
            self.right_child.upper[self.feature] = self.threshold

        self.left_child.update_bounds_below()
        self.right_child.update_bounds_below()

    def update_indicator(self):
        ''' update indicator function '''
        def is_large_enough(x):
            return np.all(
                np.array([np.greater(x[:, key], self.lower[key])
                          for key in self.lower]), axis=0
            )

        def is_small_enough(x):
            return np.all(
                np.array([np.less_equal(x[:, key], self.upper[key])
                          for key in self.upper]), axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        ''' pred function in node '''
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    ''' class that create a leaf '''
    def __init__(self, value, depth=None):
        super().__init__(depth=depth)
        self.value = value
        self.is_leaf = True

    def __str__(self):
        return f'-> leaf [value={self.value}]'

    def get_leaves_below(self):
        ''' return self as a list element '''
        return [self]

    def update_bounds_below(self):
        ''' update bounds below in leaf '''
        pass

    def pred(self, x):
        """ Returns the leaf's value. """
        return self.value


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

    def __str__(self):
        return str(self.root)

    def get_leaves(self):
        ''' return self -> root -> get leaves below '''
        return self.root.get_leaves_below()

    def update_bounds(self):
        ''' update bounds '''
        self.root.update_bounds_below()

    def update_predict(self):
        ''' Write a method that computes the prediction function '''
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([self.root.pred(x) for x in A])

    def pred(self, x):
        ''' Starts the recursion from the root '''
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        ''' fit function '''
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }
    - Accuracy on training data : { self.accuracy(self.explanatory,
                                                self.target)}""")

    def np_extrema(self, arr):
        ''' Return the minimum and maximum values of an array using NumPy'''
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        ''' Randomly selects a feature and threshold to split the node's
        subpopulation. '''
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """
        Fits a decision tree node by recursively splitting the data based on
        the best split criterion.
        """
        node.feature, node.threshold = self.split_criterion(node)

        max_criterion = np.greater(
            self.explanatory[:, node.feature],
            node.threshold)

        left_population = np.logical_and(
            node.sub_population,
            max_criterion)

        right_population = np.logical_and(
            node.sub_population,
            np.logical_not(max_criterion))

        is_left_leaf = np.any(np.array(
            [node.depth == self.max_depth - 1,
             np.sum(left_population) <= self.min_pop,
             np.unique(self.target[left_population]).size == 1]))

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = np.any(np.array(
            [node.depth == self.max_depth - 1,
             np.sum(right_population) <= self.min_pop,
             np.unique(self.target[right_population]).size == 1]))

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Create a leaf child node with the most frequent target value in the
        given subpopulation and returns the new object.
        """
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        # NOTE this should be leaf_child.subpopulation_leaf
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Create a new child node for the given parent node.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        Calculates the accuracy of the decision tree
        """
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target)) / test_target.size
