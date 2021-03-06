import re


class SimpleTokenParser:
    """
    Naive parser: kept for testing purpose
    """

    ISSUE_TAG = "<issue>"
    PATH_TAG = "<path>"
    ENTITY_NAME = "<entity>"
    FUNCTION_TAG = "<function>"
    NUMBER_TAG = "<number>"
    LANGUAGE_TAG = "<language>"

    def __init__(self):
        self.issue = re.compile("^[a-zA-Z]+[\-]?[0-9]+$")
        self.abbreviations = {"url", "ci", "raii", "bau", "slo", "api", "stl"}
        self.languages = {"c++", "mef", "java", "c", "cpp", "ant", "groovy", "js", "scala"}

    def parse(self, token: str) -> str:
        if token.lower() in self.abbreviations:
            return token

        if token.lower() in self.languages:
            return self.LANGUAGE_TAG

        if self.issue.match(token):
            return self.ISSUE_TAG

        if all(c.isdigit() for c in token):
            return self.NUMBER_TAG

        if token.isupper():
            return self.ENTITY_NAME

        if "_" in token:
            return self.ENTITY_NAME if token.isupper() else self.FUNCTION_TAG

        if self.count(token, lambda c: c == "/") >= 2:
            return self.PATH_TAG

        if self.count(token[1:], lambda c: c.isupper()) >= 1:
            return self.FUNCTION_TAG

        return token.lower()

    @staticmethod
    def count(token, pred):
        return sum(1 if pred(c) else 0 for c in token)


class TokenParser:
    ISSUE_TAG = "<issue>"
    PATH_TAG = "<path>"
    ENTITY_NAME = "<entity>"
    FUNCTION_TAG = "<function>"
    CLASS_TAG = "<class>"
    NUMBER_TAG = "<number>"
    LANGUAGE_TAG = "<language>"

    def __init__(self):
        self.issue = re.compile("^[a-zA-Z]+[\-]?[0-9]+$")
        self.abbreviations = {"url", "ci", "raii", "bau", "slo", "api", "stl", "owners"}
        self.languages = {"c++", "mef", "java", "c", "cpp", "ant", "groovy", "js", "scala"}
        self.valid_function_start = set("abcdefghijklmnopqrstuvwxyz_")
        self.valid_package_characters = set("abcdefghijklmnopqrstuvwxyz_-.")

    def generate(self, parsed: str) -> str:
        # TODO - do better
        if parsed == self.ISSUE_TAG:
            return "FIX-ME-12"
        if parsed == self.PATH_TAG:
            return "lib/folder1/folder2"
        if parsed == self.ENTITY_NAME:
            return "PROJECT_TOTO"
        if parsed == self.FUNCTION_TAG:
            return "getImpactedYield"
        if parsed == self.CLASS_TAG:
            return "CollateralAgreement"
        if parsed == self.NUMBER_TAG:
            return "123"
        if parsed == self.LANGUAGE_TAG:
            return "C++"
        return parsed

    def parse(self, token: str) -> str:
        if all(c.isdigit() for c in token):
            return self.NUMBER_TAG

        if token.lower() in self.abbreviations:
            return token

        if token.lower() in self.languages:
            return self.LANGUAGE_TAG

        if self.issue.match(token):
            return self.ISSUE_TAG

        if token.isupper():
            return self.ENTITY_NAME

        if self.is_path(token):
            return self.PATH_TAG

        start = token.rfind('.')
        if start != -1:
            if start + 1 < len(token) and self.is_package_name(token[:start]):
                if self.is_unqualified_function(token[start+1:], with_namespace=True):
                    return self.FUNCTION_TAG
                if self.is_unqualified_class(token[start+1], with_namespace=True):
                    return self.CLASS_TAG
        else:
            if self.is_unqualified_function(token):
                return self.FUNCTION_TAG
            if self.is_unqualified_class(token):
                return self.CLASS_TAG

        return token.lower()

    @classmethod
    def is_path(cls, token):
        return cls.count(token, lambda c: c == "/") >= 2

    def is_package_name(self, token: str) -> bool:
        return all(c in self.valid_package_characters for c in token)

    def is_unqualified_function(self, token: str, with_namespace=False) -> bool:
        if not token:
            return False
        if not token[0] in self.valid_function_start:
            return False
        if "_" in token[1:] and token.islower():
            return True
        return with_namespace or self.count(token[1:], lambda c: c.isupper()) >= 1

    def is_unqualified_class(self, token: str, with_namespace=False) -> bool:
        if not token:
            return False
        if not token[0].isupper() or not token.isalnum():
            return False
        return with_namespace or self.count(token[1:], lambda c: c.isupper()) >= 1

    @staticmethod
    def count(token, pred):
        return sum(1 if pred(c) else 0 for c in token)
