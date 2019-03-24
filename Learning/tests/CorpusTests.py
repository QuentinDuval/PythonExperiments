from Learning.Corpus import *

import unittest


class CommmitMessageCorpusTest(unittest.TestCase):

    def test_target_class_refactor(self):
        res, fix = CommitMessageCorpus.match_fix("{PROJECT}[REFAC]: Use a proxy to sync all the views")
        self.assertEqual(res, CommitMessageCorpus.REFACTOR)
        self.assertEqual(fix, "{PROJECT}: Use a proxy to sync all the views")

        res, fix = CommitMessageCorpus.match_fix("{PROJECT}[REFACTOR]: cleaning and refactoring")
        self.assertEqual(res, CommitMessageCorpus.REFACTOR)
        self.assertEqual(fix, "{PROJECT}: cleaning and refactoring")

        res, fix = CommitMessageCorpus.match_fix("{PROJECT}[REFACTOR](fixtures): cleaning and refactoring")
        self.assertEqual(res, CommitMessageCorpus.REFACTOR)
        self.assertEqual(fix, "{PROJECT}(fixtures): cleaning and refactoring")

    def test_target_class_fix(self):
        res, fix = CommitMessageCorpus.match_fix("{PROJECT} [FIX] refactor and fix the blablabla")
        self.assertEqual(res, CommitMessageCorpus.FIX)
        self.assertEqual(fix, "{PROJECT}  refactor and fix the blablabla")

        res, fix = CommitMessageCorpus.match_fix(
            "[fix] [CMAKE ISSUES] (unit tests): Shorten the length of the unit test path as CMake has trouble dealing with long paths.")
        self.assertEqual(res, CommitMessageCorpus.FIX)
        self.assertEqual(fix, "[CMAKE ISSUES] (unit tests): Shorten the length of the unit test path as CMake has trouble dealing with long paths.")

    def test_target_class_feat(self):
        res, fix = CommitMessageCorpus.match_fix("{PROJECT} [FEAT] (fixing screen) improving the fixing screen")
        self.assertEqual(res, CommitMessageCorpus.FEAT)
        self.assertEqual(fix, "{PROJECT}  (fixing screen) improving the fixing screen")

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
