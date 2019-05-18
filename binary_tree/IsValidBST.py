"""
https://practice.geeksforgeeks.org/problems/check-for-bst/1
"""


"""
#include <functional>
using Pred = std::function<bool(int)>;

bool isValid(Node* root, Pred lo, Pred hi)
{
    if (!root) return true;
    
    int val = root->data;
    return lo(val)
        && hi(val)
        && isValid(root->left, lo, [=](int i) { return i <= val; })
        && isValid(root->right, [=](int i) { return i >= val; }, hi);
}

bool isBST(Node* root)
{
    return isValid(root,
        [](int i) { return true; },
        [](int i) { return true; });
}
"""
