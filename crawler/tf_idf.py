from crawler.parsing import NltkTokenizer, list_files_in_folder

from collections import defaultdict
import os

# TODO - classify document based on TF-IDF


class PostingList:
    def __init__(self):
        self.total = 0
        self.postings = defaultdict(int)

    def add_document(self, document):
        self.total += 1
        self.postings[document] += 1

    def __repr__(self):
        return repr(self.postings)


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(PostingList)
        self.tokenizer = NltkTokenizer()

    def add_document(self, file_path: str):
        folder_name, file_name = os.path.split(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                for word in self.tokenizer.tokenize(line):
                    self.index[word].add_document(file_name)


inv_index = InvertedIndex()
for file_path in list_files_in_folder("posts/fluentcpp"):
    inv_index.add_document(file_path)

for word, posting in sorted(inv_index.index.items(), key=lambda p: p[1].total, reverse=True):
    print(word)
    print(posting)

