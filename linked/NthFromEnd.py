"""
https://practice.geeksforgeeks.org/problems/nth-node-from-end-of-linked-list/1
"""


"""
int getNthFromLast(Node *head, int n)
{
    auto last = head;
    for (int i = 0; i < n; ++i) {
        if (!last) return -1;
        last = last->next;
    }

    auto curr = head;
    for (; last; last = last->next)
        curr = curr->next;
    return curr->data;
}
"""
