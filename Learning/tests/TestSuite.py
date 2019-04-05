from Learning.tests.CorpusTests import *
from Learning.tests.TokenizerTests import *
from Learning.tests.TokenParserTests import *


suite = unittest.TestSuite()
suite.addTest(CorpusTests())
suite.addTest(TokenizerTests())
suite.addTest(TokenParserTests())
unittest.main()
