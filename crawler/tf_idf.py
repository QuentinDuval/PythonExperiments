from typing import List

from crawler.parsing import NltkTokenizer, list_files_in_folder

from collections import defaultdict
import os


# TODO - use Whoosh (indexing in pure Python): https://whoosh.readthedocs.io/en/latest/quickstart.html
# TODO - how to extract the significant terms: https://whoosh.readthedocs.io/en/latest/keywords.html

class PostingList:
    """
    Posting list for a given word:
    - Keep a document to number of occurrence number dictionary
    - Keep the total amount of time a word has been seen
    """

    def __init__(self):
        self.total = 0
        self.postings = defaultdict(int)

    def add_posting(self, document):
        self.total += 1
        self.postings[document] += 1

    def __repr__(self):
        return repr(self.postings)


class InvertedIndex:
    """
    Maintains an inverted index of documents (blog posts here) from words to documents that contain the words.
    """

    def __init__(self):
        self.documents = defaultdict(set)
        self.document_postings = defaultdict(PostingList)
        self.next_word_postings = defaultdict(PostingList)
        self.tokenizer = NltkTokenizer()

    def add_document(self, file_path: str):
        folder_name, file_name = os.path.split(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                prev_word = None
                for word in self.tokenizer.tokenize(line):
                    self.documents[file_name].add(word)
                    self.document_postings[word].add_posting(file_name)
                    if prev_word is not None:
                        self.next_word_postings[prev_word].add_posting(word)
                    prev_word = word

    def get_topic(self, document_title: str) -> List[str]:
        """
        Return the titles of the document that are the most related to the document_title
        """
        # TODO
        #   - get the words in the document, and look at the words in the posting list of documents
        #   - check if the document is among the most representative for each word (and keep this word)
        pass

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

    def get_most_common_next_words(self) -> List[str]:
        pass


inv_index = InvertedIndex()
for file_path in list_files_in_folder("posts/fluentcpp"):
    inv_index.add_document(file_path)

for i, (word, posting) in enumerate(sorted(inv_index.document_postings.items(), key=lambda p: p[1].total, reverse=True)):
    if i < 10:
        print(word)
        print(posting)

