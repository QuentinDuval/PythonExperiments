import unittest

from Learning.TokenParser import *


class TokenParserTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenParser()

    def parse(self, token):
        return self.tokenizer.parse(token)

    def test_number_recognition(self):
        self.assertEqual(self.tokenizer.NUMBER_TAG, self.parse("1234"))

    def test_language_recognition(self):
        self.assertEqual(self.tokenizer.LANGUAGE_TAG, self.parse("C++"))
        self.assertEqual(self.tokenizer.LANGUAGE_TAG, self.parse("cpp"))
        self.assertEqual(self.tokenizer.LANGUAGE_TAG, self.parse("Java"))
        self.assertEqual(self.tokenizer.LANGUAGE_TAG, self.parse("js"))

    def test_jira_recognition(self):
        self.assertEqual(self.tokenizer.ISSUE_TAG, self.parse("FIX-128"))
        # self.assertEqual(self.tokenizer.ISSUE_TAG, self.parse("FIX-ME-128"))

    def test_entity_recognition(self):
        self.assertEqual(self.tokenizer.ENTITY_NAME, self.parse("ENT"))
        self.assertEqual(self.tokenizer.ENTITY_NAME, self.parse("SUB_ENT"))
        self.assertEqual(self.tokenizer.ENTITY_NAME, self.parse("SUB_ENT_ID"))
        self.assertEqual(self.tokenizer.ENTITY_NAME, self.parse("SUB-ENT"))

    def test_path_recognition(self):
        self.assertEqual(self.tokenizer.PATH_TAG, self.parse("include/hello/world"))
        self.assertEqual(self.tokenizer.PATH_TAG, self.parse("/hello/world"))

    def test_function_recognition(self):
        self.assertEqual(self.tokenizer.FUNCTION_TAG, self.parse("helloWorld"))
        self.assertEqual(self.tokenizer.FUNCTION_TAG, self.parse("hello_world"))

    def test_qualified_function_recognition(self):
        self.assertEqual(self.tokenizer.FUNCTION_TAG, self.parse("ns::sub_ns::helloWorld"))
        self.assertEqual(self.tokenizer.FUNCTION_TAG, self.parse("ns::sub_ns::hello_world"))
        # self.assertEqual(self.tokenizer.FUNCTION_TAG, self.parse("std::transform"))

    def test_class_recognition(self):
        self.assertEqual(self.tokenizer.CLASS_TAG, self.parse("ClassName"))

    def test_qualified_class_recognition(self):
        pass
        # self.assertEqual(self.tokenizer.CLASS_TAG, self.parse("pack.package.ClassName"))
        # self.assertEqual(self.tokenizer.CLASS_TAG, self.parse("pack.sub-pack.ClassName"))
