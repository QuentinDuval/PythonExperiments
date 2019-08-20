# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, head1, head2):
        l1 = self.length(head1)
        l2 = self.length(head2)
        if l1 < l2:
            l2, l1 = l1, l2
            head2, head1 = head1, head2

        for _ in range(l1 - l2):
            head1 = head1.next

        while head1 and head2:
            if head1 == head2:
                return head1
            else:
                head1 = head1.next
                head2 = head2.next
        return None

    def length(self, head):
        count = 0
        while head:
            count += 1
            head = head.next
        return count
