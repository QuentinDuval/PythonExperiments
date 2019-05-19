"""
https://practice.geeksforgeeks.org/problems/check-for-balanced-tree/1

Given a binary tree, find if it is height balanced or not.
A tree is height balanced if difference between heights of left and right subtrees is not more than one
for all nodes of tree.

Note that collecting the heights of the leafs and checking their height would not work...
"""

"""
#include <utility>

std::pair<bool, int> get_balanced_depth(Node* node)
{
    if (!node)
        return { true, 0 };

    auto left = get_balanced_depth(node->left);
    if (!left.first)
        return { false, 0 };

    auto right = get_balanced_depth(node->right);
    if (!right.first)
        return { false, 0 };

    auto max_depth = max(left.second, right.second);
    if (max_depth > 1 + min(left.second, right.second))
        return { false, 0 };

    return { true, 1 + max_depth };
}

bool isBalanced(Node *root)
{
    return get_balanced_depth(root).first;
}
"""
