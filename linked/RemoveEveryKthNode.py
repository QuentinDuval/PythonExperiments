"""
https://practice.geeksforgeeks.org/problems/remove-every-kth-node/1
"""

"""
// Iterate with previous and current pointer
Node* deleteK(Node* head, int k)
{
    if (k == 0) {
        return head;
    }
    
    if (k == 1) {
        while (head) {
            Node* curr = head;
            head = head->next;
            delete curr;
        }
        return head;
    }
    
    Node* prev = nullptr;
    Node* curr = head;
    
    int count = 1;
    while (curr) {
        if (count == k) {
            prev->next = curr->next;
            delete curr;
            curr = prev->next;
            count = 1;
        }
        else {
            prev = curr;
            curr = curr->next;
            count += 1;
        }
    }
    return head;
}

"""
