"""
https://practice.geeksforgeeks.org/problems/rotate-a-linked-list/1
"""


"""
void rotate(node **head_ref, int k)
{
    // You need to know the size to reduce 'k' with modulo
    // So the best is to compute the size and find the 'tail' in one pass
    // Then use this tail to add the first 'k' elements to the end
    if (!head || !*head_ref) return;

    auto tail = *head_ref;
    int length = 1;
    while (tail->next) {
        length += 1;
        tail = tail->next;
    }

    k = k % length;
    for (int i = 0; i < k; ++i) {
        auto head = *head_ref;
        *head_ref = head->next;
        tail->next = head;
        tail = head;
        head->next = nullptr;
    }
}
"""
