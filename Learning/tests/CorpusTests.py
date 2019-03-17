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

        res, fix = CommitMessageCorpus.match_fix("{COL_BAU}[REFACTOR](MRV fixtures): cleaning and refactoring")
        self.assertEqual(res, CommitMessageCorpus.REFACTOR)
        self.assertEqual(fix, "{COL_BAU}(MRV fixtures): cleaning and refactoring")

    def test_target_class_fix(self):
        pass # TODO - this and other classes as well

    def test_index_vs_label(self):
        for target in CommitMessageCorpus.TARGET_CLASSES:
            target_index = CommitMessageCorpus.target_class_index(target)
            target_label = CommitMessageCorpus.target_class_label(target_index)
            self.assertEqual(target, target_label)

    '''
    def test_real_data_set(self):
        test_corpus = CommitMessageCorpus.from_file('../resources/perforce_cl_test.txt')    # TODO - use different
        for fix, target in test_corpus:
            if fix == "{COL_BAU}[REFACTOR]: cleaning and refactoring":
                self.assertEqual(CommitMessageCorpus.REFACTOR, target)
    '''
