"""
You are given a list of strings constructed this way:

Initially the strings were sorted lexicographically (example: ["abc", "acad"])
Then these string went through a bijective mapping (example: a->r, b->h, c->x, d->a)
You are given these transformed strings, ordered as they were before the bijective mapping.

Find a mapping that satisfies the constraints.

Note: by the way, it is easy to test, just apply the reversed mapping and the collection you get should be sorted.
"""


import numpy as np
import string
from typing import *


def topological_sort(graph: Dict[Any, List[Any]]) -> List[Any]:
    ordered = []
    discovered = set()

    def dfs(node):
        for children in graph.get(node, []):
            if children not in discovered:
                discovered.add(children)
                dfs(children)
        ordered.append(node)

    for start_node in graph.keys():
        if start_node not in discovered:
            discovered.add(start_node)
            dfs(start_node)
    return list(reversed(ordered))


def guess_mapping(words: List[str]) -> Dict[str, str]:
    n = len(words)
    ordering_constraints = {c: set() for c in string.ascii_lowercase}

    # TODO - you should be able to do it in O(N):
    #   - scan first letters (to have the ordering there)
    #   - scan second letters in each group having same first letter
    #   - etc => can basically be done with a recursion

    for i in range(n):
        for j in range(i+1, n):
            w1 = words[i]
            w2 = words[j]
            m = min(len(w1), len(w2))
            for k in range(m):
                if w1[k] != w2[k]:
                    ordering_constraints[w1[k]].add(w2[k])
                    break

            # Quick optimization to let transitivity in the graph do its work (could be done for letter 2, etc.)
            if w1[0] != w2[0]:
                break

    ordered_letters = topological_sort(ordering_constraints)
    return {dst: src for src, dst in zip(string.ascii_lowercase, ordered_letters)}


def apply_mapping(words: List[str], mapping: List[str]) -> List[str]:
    previous = []
    for word in words:
        prev = ""
        for c in word:
            prev += mapping[c]
        previous.append(prev)
    return previous


"""
Tests
"""


def generative_testing(nb_test_cases: int, sentence_size: int, word_min_size: int, word_max_size: int):
    letter_indexes = list(range(len(string.ascii_lowercase)))

    for _ in range(nb_test_cases):

        # Generate sentence
        sentence = []
        for _ in range(sentence_size):
            word_size = np.random.randint(word_min_size, word_max_size+1)
            word = "".join(string.ascii_lowercase[i] for i in np.random.choice(letter_indexes, size=word_size))
            sentence.append(word)
        sentence.sort()

        # Generate mapping
        mapping = list(string.ascii_lowercase)
        np.random.shuffle(mapping)
        mapping = {src: dst for src, dst in zip(string.ascii_lowercase, mapping)}
        sentence = apply_mapping(sentence, mapping)

        # Test the reversal of the mapping
        rev_mapping = guess_mapping(sentence)
        output = apply_mapping(sentence, rev_mapping)
        assert output == list(sorted(output))


'''
words = "da abc abad bc".split(" ")
mapping = guess_mapping(words)

print(mapping)
# outputs {'d': 'a', 'c': 'b', 'a': 'c', 'b': 'd'}

print(apply_mapping(words, mapping))
# outputs ['ac', 'cdb', 'cdca', 'db']
'''

generative_testing(nb_test_cases=100, sentence_size=15, word_min_size=2, word_max_size=5)
