"""
#include <vector>
#include <utility>

int height(Node* node)
{
    int max_height = 0;
    std::vector<std::pair<Node*, int>> to_visit;
    if (node)
        to_visit.push_back({node, 1});

    while (!to_visit.empty())
    {
        auto top = to_visit.back();
        to_visit.pop_back();
        auto node = top.first;
        auto height = top.second;
        max_height = max(max_height, height);
        if (node->left)
            to_visit.push_back({node->left, height+1});
        if (node->right)
            to_visit.push_back({node->right, height+1});
    }

    return max_height;
}
"""
