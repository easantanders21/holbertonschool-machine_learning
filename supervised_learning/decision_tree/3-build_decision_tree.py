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
        leaves_list = []
        if self.left_child:
            if self.left_child.is_leaf:
                leaves_list.append(f"-> leaf [value={self.left_child.value}]")
            else:
                leaves_list.extend(self.left_child.get_leaves_below())
        if self.right_child:
            if self.right_child.is_leaf:
                leaves_list.append(f"-> leaf [value={self.right_child.value}]")
            else:
                leaves_list.extend(self.right_child.get_leaves_below())
        return leaves_list


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
