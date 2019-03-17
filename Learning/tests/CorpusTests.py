from Learning.Corpus import *

import unittest


class CommmitMessageCorpusTest(unittest.TestCase):

    def test_target_class_refactor(self):
        res, fix = CommitMessageCorpus.match_fix("{COL_BAU}[REFAC]:  Use a proxy to sync all FitNesse views")
        self.assertEqual(res, CommitMessageCorpus.REFACTOR)
        self.assertEqual(fix, "{COL_BAU}:  Use a proxy to sync all FitNesse views")

        res, fix = CommitMessageCorpus.match_fix("{COL_BAU}[REFACTOR]: cleaning and refactoring")
        self.assertEqual(res, CommitMessageCorpus.REFACTOR)
        self.assertEqual(fix, "{COL_BAU}: cleaning and refactoring")

