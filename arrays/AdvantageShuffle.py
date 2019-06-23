"""
https://leetcode.com/problems/advantage-shuffle/

Given two arrays A and B of equal size, the advantage of A with respect to B is the number of indices i
for which A[i] > B[i].

Return any permutation of A that maximizes its advantage with respect to B.
"""

from typing import List


class Solution:
    def advantageCount(self, A: List[int], B: List[int]) -> List[int]:
        """
        Any permutation of A means that as long as we have the same elements, we are good.
        So we can summarize A as a set of integers we pick from.

        Then we can try GREEDY:
        - For each element of B, pick the smallest element of A that is higher
        - Otherwise, pick the smallest element of A (does not matter, we have bigger)
        - It works because ???
        We can do this using a sorted MULTI-set and complexity is O(N log N)
        """

        '''
        In C++ (172ms):
        
        vector<int> advantageCount(vector<int> const& as, vector<int> const& bs)
        {
            vector<int> permutation;
            multiset<int> reserve(as.begin(), as.end());
            for (int i = 0; i < bs.size(); ++i)
            {
                int b = bs[i];
                auto chosen = reserve.upper_bound(b);
                if (chosen == reserve.end())
                    chosen = reserve.begin();
                permutation.push_back(*chosen);
                reserve.erase(chosen);
            }
            return permutation;
        }
        '''

        '''
        In Java (65ms)
        
        public int[] advantageCount(int[] A, int[] B) {
            TreeMap<Integer, Integer> reserve = new TreeMap<>();
            for (int a: A)
                reserve.put(a, reserve.getOrDefault(a, 0) + 1);
            
            int[] answer = new int[A.length];
            for (int i = 0; i < B.length; ++i)
            {
                int b = B[i];
                Map.Entry<Integer, Integer> a = reserve.higherEntry(b);
                if (a == null)
                    a = reserve.firstEntry();
                
                answer[i] = a.getKey();
                if (a.getValue() == 1)
                    reserve.remove(a.getKey());
                else
                    reserve.put(a.getKey(), a.getValue() - 1);
            }
            return answer;
        }
        '''

        """
        Sorting would also help us: we could sort both A and B, but since we have
        to return the permutation, we could lose the ordering of B...
        
        But we can instead sort the tables based on the indices.
        
        REMEMBER: YOU DO NOT NEED MAP WHEN ALL KEYS ARE INSERTED THEN READ (EXCEPT IF REMOVED).

        Or we could also do 2 heaps:
        - min heap for elements of A ordered by value
        - min heap for elements of B ordered by (value, index)
        Then if A.top() is smaller than B.top(), reserve it for later (end of the list)
        But if A.top() is higher than B.top(), put it at the index place, and pop both
        
        This implementation beats 90%!
        """

        A.sort()
        indices = list(range(len(B)))
        indices.sort(key=lambda i: B[i])

        ia = 0
        ib = 0
        answer = [0] * len(B)
        for _ in range(len(B)):
            if A[ia] > B[indices[ib]]:
                answer[indices[ib]] = A[ia]
                ia += 1
                ib += 1
            else:
                answer[indices[-1]] = A[ia]
                ia += 1
                indices.pop()
        return answer

