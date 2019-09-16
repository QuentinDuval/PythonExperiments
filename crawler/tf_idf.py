from typing import List

from crawler.parsing import NltkTokenizer, list_files_in_folder

from collections import defaultdict
import os


class PostingList:
    """
    Posting list for a given word:
    - Keep a document to number of occurrence number dictionary
    - Keep the total amount of time a word has been seen
    """

    def __init__(self):
        self.total = 0
        self.postings = defaultdict(int)

    def add_document(self, document):
        self.total += 1
        self.postings[document] += 1

    def __repr__(self):
        return repr(self.postings)


class InvertedIndex:
    """
    Maintains an inverted index of documents (blog posts here) from words to documents that contain the words.
    """

    def __init__(self):
        self.posting_lists = defaultdict(PostingList)
        self.tokenizer = NltkTokenizer()

    def add_document(self, file_path: str):
        folder_name, file_name = os.path.split(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                for word in self.tokenizer.tokenize(line):
                    self.posting_lists[word].add_document(file_name)

    def get_topic(self, document_title: str) -> List[str]:
        """
        Return the titles of the document that are the most related to the document_title
        """
        pass  # TODO

    def get_documents(self, words: List[str]) -> List[str]:
        """
        Return the list of title of the document that are the most related to the words given as input
        """
        pass  # TODO

    def get_common_expressions(self) -> List[str]:
        """
        Use TF-IDF / correlation measure to return the terms that are the most associated together
        """
        pass  # TODO


inv_index = InvertedIndex()
for file_path in list_files_in_folder("posts/fluentcpp"):
    inv_index.add_document(file_path)

for i, (word, posting) in enumerate(sorted(inv_index.posting_lists.items(), key=lambda p: p[1].total, reverse=True)):
    if i < 10:
        print(word)
        print(posting)

