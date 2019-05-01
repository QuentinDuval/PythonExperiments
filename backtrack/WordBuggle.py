"""
https://practice.geeksforgeeks.org/problems/word-boggle/0

Given a dictionary, a method to do lookup in dictionary and a M x N board where every cell has one character.
Find all possible words that can be formed by a sequence of adjacent characters.
Note that we can move to any of 8 adjacent characters, but a word should not have multiple instances of same cell.
"""


def reshape(m, n, letters):
    matrix = []
    for i in range(m):
        matrix.append(letters[i * n:i * n + n])
    return matrix


def word_buggle(words, m, n, letters):
    found = set()
    matrix = reshape(m, n, letters)

    # We use the sorted list as a "trie" to search for prefixes
    words.sort()

    def search_prefix(prefix):
        lo = 0
        hi = len(words) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if prefix == words[mid]:
                return words[mid]
            if prefix < words[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        if lo == len(words) or not words[lo].startswith(prefix):
            return None
        return words[lo]

    def neighbors(i, j):
        if i > 0:
            yield (i-1, j)
        if j > 0:
            yield (i, j-1)
        if i > 0 and j > 0:
            yield (i-1, j-1)
        if i < m - 1:
            yield (i+1, j)
        if j < n - 1:
            yield (i, j+1)
        if i < m - 1 and j < n - 1:
            yield (i+1, j+1)
        if i > 0 and j < n - 1:
            yield (i-1, j+1)
        if j > 0 and i < m - 1:
            yield (i+1, j-1)

    def word_buggle_dfs(stack):
        # Pruning based on search in a "trie"
        prefix = "".join(matrix[i][j] for i, j in stack)
        word = search_prefix(prefix)
        if not word:
            return

        if prefix == word:
            found.add(word)

        i, j = stack[-1]
        for adj in neighbors(i, j):
            if adj not in stack:
                stack.append(adj)
                word_buggle_dfs(stack)
                stack.pop()

    def word_buggle_from(i, j):
        word_buggle_dfs([(i, j)])

    for i in range(m):
        for j in range(n):
            word_buggle_from(i, j)
    return found


# '''
res = word_buggle(
    words="GEEKS FOR QUIZ GO".split(),
    m=3, n=3,
    letters="G I Z U E K Q S E".split()
)
print(res)
# Expected { "GEEKS", "QUIZ" }
# '''
