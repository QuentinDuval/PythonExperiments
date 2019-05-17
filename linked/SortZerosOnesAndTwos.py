"""
https://practice.geeksforgeeks.org/problems/given-a-linked-list-of-0s-1s-and-2s-sort-it/1

Given a linked list of size N consisting of 0s, 1s and 2s. The task os to sort this linked list such that all zeroes
segregate to headside, 2s at the end and 1s in the mid of 0s and 2s.


void sortList(Node *head)
{
    // First solution is to keep a pointer to head and tail for 0s, 1s and 2s

    // Second solution is to keep a pointer to heads only:
    // - never mind the order or the node (not stable)
    // - connect at the end (we could probably do during)

    /*
    Node* parts[3] = { nullptr, nullptr, nullptr };
    while (head) {
        Node* curr = head;
        head = head->next;
        curr->next = parts[curr->data];
        parts[curr->data] = curr;
    }

    for (int i = 1; i >= 0; ++i) {
        if (parts[i] == nullptr) {
            parts[i] = parts[i+1];
        } else {
            Node* last = parts[i];
            while (last->next)
                last = last->next;
            last->next = parts[i+1];
        }
    }
    */

    // Last solution is to sort the values (just count the 0, 1, 2)
    // Then do a second pass to switch the values

    std::vector<int> counts(3, 0);
    for (Node* curr = head; curr; curr = curr->next) {
        counts[curr->data] += 1;
    }

    for (Node* curr = head; curr; curr = curr->next) {
        for (int i = 0; i < counts.size(); ++i) {
            if (counts[i] > 0) {
                counts[i] -= 1;
                curr->data = i;
                break;
            }
        }
    }
}
"""

# TODOs
