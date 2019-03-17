import functools
import re

from Learning.Utils import *


class CommitMessageCorpus:
    REFACTOR = "refactor"
    FEAT = "feat"
    REVERT = "revert"
    FIX = "fix"
    TARGET_CLASSES = [REFACTOR, FEAT, REVERT, FIX]

    def __init__(self, xs, ys, unclassified=None):
        self.xs = xs
        self.ys = ys
        self.unclassified = unclassified or []

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]

    def get_inputs(self):
        return self.xs

    def get_targets(self):
        return self.ys

    def get_unclassified(self):
        return self.unclassified

    @classmethod
    def target_class_index(cls, label):
        return cls.TARGET_CLASSES.index(label)

    @classmethod
    def target_class_label(cls, class_index):
        return cls.TARGET_CLASSES[class_index]

    def split(self, ratio):
        lhs, rhs = join_split(self.xs, self.ys, ratio)
        return CommitMessageCorpus(*lhs), CommitMessageCorpus(*rhs)

    @staticmethod
    def match_fix(fix_description):
        for target_class in ["revert", "fix", "feat", "refactor"]:
            matcher = get_target_matcher(target_class)
            match = matcher(fix_description)
            if match:
                return target_class, match
        return None, fix_description

    @staticmethod
    def read_manual_exceptions():
        with open('resources/manual_exceptions.txt', 'r') as f:
            exceptions = {}
            commit_description = None
            for line in f:
                if commit_description is None:
                    commit_description = line.strip()
                else:
                    exceptions[commit_description] = line.strip().lower()
                    commit_description = None
            return exceptions

    @classmethod
    def from_file(cls, file_name, keep_unclassified=False):
        xs = []
        ys = []
        unclassified = []
        manual_exceptions = cls.read_manual_exceptions()
        with open(file_name, 'r') as inputs:
            for fix_description in inputs:
                fix_description = fix_description.strip()
                if fix_description in manual_exceptions:
                    print("Manual exception", fix_description)
                    xs.append(fix_description)
                    ys.append(manual_exceptions[fix_description])
                else:
                    target_class, fix_description = cls.match_fix(fix_description)
                    if target_class:
                        xs.append(fix_description)
                        ys.append(target_class)
                    elif keep_unclassified:
                        unclassified.append(fix_description)
        return cls(xs, ys, unclassified)

    @classmethod
    def from_split(cls, split_name):
        file_path = 'resources/perforce_cl_test.txt' if split_name == 'test' else 'resources/perforce_cl_train.txt'
        return cls.from_file(file_path)


class Matcher:
    def __init__(self):
        self.regex_list = []

    def add(self, regex, with_surroundings=True):
        surroundings = "[a-zA-Z0-9_\-+ ]*"
        if with_surroundings:
            regex = surroundings + self.to_case_insensitive(regex) + surroundings
        else:
            regex = self.to_case_insensitive(regex)
        self.regex_list.append(re.compile("\[" + regex + "\]"))
        self.regex_list.append(re.compile("\(" + regex + "\)"))
        self.regex_list.append(re.compile("\{" + regex + "\}"))

    def match(self, fix_description):
        for regex in self.regex_list:
            if regex.search(fix_description):
                return regex.sub("", fix_description).lstrip()
        return None

    def __call__(self, fix_description):
        return self.match(fix_description)

    @staticmethod
    def to_case_insensitive(regex):
        out = ""
        for c in regex:
            out += "[" + c.lower() + c.upper() + "]"
        return out


@functools.lru_cache(maxsize=None)
def get_target_matcher(target_class):
    matcher = Matcher()
    if target_class == "revert":
        matcher.add("revert")
    elif target_class == "refactor":
        matcher.add("refac")
        matcher.add("clean")
        matcher.add("ptech")
        matcher.add("enabler")
        matcher.add("constification")
        matcher.add("technical")  # TODO - might not be good data
        matcher.add("style")
        # matcher.add("test") # TODO - might not be a good idea
    elif target_class == "fix":
        matcher.add("bugfix")
        matcher.add("mlk")
        matcher.add("fix", with_surroundings=False)
        matcher.add("bug")
        matcher.add("cwe")
        matcher.add("regression")
    elif target_class == "feat":
        matcher.add("feat")
        matcher.add("enhancement")
        # matcher.add("diagnostic") # TODO - definitively not good data
        matcher.add("perf")
        # matcher.add("optim") # TODO - might not be good data
    return matcher
