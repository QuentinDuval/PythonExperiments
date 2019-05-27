"""
https://leetcode.com/problems/accounts-merge/

Given a list accounts, each element accounts[i] is a list of strings, where the first element accounts[i][0] is a name, and the rest of the elements are emails representing emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there is some email that is common to both accounts. Note that even if two accounts have the same name, they may belong to different people as people could have the same name. A person can have any number of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element of each account is the name, and the rest of the elements are emails in sorted order. The accounts themselves can be returned in any order.
"""


from collections import defaultdict
from typing import List


class Merger:
    def __init__(self):
        self.edges = defaultdict(list)

    def add(self, mails):
        for mail in mails:
            if mail not in self.edges:
                self.edges[mail] = []
        for mail1, mail2 in zip(mails[:-1], mails[1:]):
            self.edges[mail1].append(mail2)
            self.edges[mail2].append(mail1)

    def groups(self):
        discovered = set()
        for start in self.edges.keys():
            if start in discovered:
                continue

            group = set()
            discovered.add(start)
            to_visit = [start]
            while to_visit:
                node = to_visit.pop()
                group.add(node)
                for n in self.edges[node]:
                    if n not in discovered:
                        discovered.add(n)
                        to_visit.append(n)
            yield group


class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        by_name = defaultdict(Merger)
        for account in accounts:
            name = account[0]
            mails = account[1:]
            by_name[name].add(mails)

        merged = []
        for name, merger in by_name.items():
            for group in merger.groups():
                res = [name]
                res.extend(sorted(group))
                merged.append(res)
        return merged
