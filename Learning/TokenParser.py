import re


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
        self.abbreviations = {"url", "ci", "raii", "bau", "slo", "api", "stl"}
        self.languages = {"c++", "mef", "java", "c", "cpp", "ant", "groovy", "js", "scala"}

    def parse(self, token: str) -> str:
        if all(c.isdigit() for c in token):
            return self.NUMBER_TAG

        if token.lower() in self.abbreviations:
            return token

        if token.lower() in self.languages:
            return self.LANGUAGE_TAG

        # TODO - improve this
        if self.issue.match(token):
            return self.ISSUE_TAG

        if token.isupper():
            return self.ENTITY_NAME

        # TODO - improve (typically if start with / it is obvious)
        if self.count(token, lambda c: c == "/") >= 2:
            return self.PATH_TAG

        # TODO - make function name more... exact
        if "_" in token and token.islower():
            return self.FUNCTION_TAG
        if self.count(token[1:], lambda c: c.isupper()) >= 1:
            return self.FUNCTION_TAG if token[0].islower() else self.CLASS_TAG

        return token.lower()

    @staticmethod
    def count(token, pred):
        return sum(1 if pred(c) else 0 for c in token)
